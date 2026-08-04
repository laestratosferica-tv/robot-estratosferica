from __future__ import annotations

import hashlib
import hmac
import html
import json
import os
import secrets
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Mapping

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse


AUTH_URL = "https://www.tiktok.com/v2/auth/authorize/"
TOKEN_URL = "https://open.tiktokapis.com/v2/oauth/token/"
USER_INFO_URL = "https://open.tiktokapis.com/v2/user/info/"
INBOX_INIT_URL = "https://open.tiktokapis.com/v2/post/publish/inbox/video/init/"
SESSION_COOKIE = "estratosferica_creator_session"
MAX_VIDEO_BYTES = 50 * 1024 * 1024
ALLOWED_VIDEO_TYPES = {"video/mp4", "video/quicktime"}


def _enabled(environment: Mapping[str, str], name: str) -> bool:
    return environment.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    client_key: str
    client_secret: str
    redirect_uri: str
    session_secret: str
    public_base_url: str
    draft_transfer_enabled: bool
    sandbox_review_mode: bool
    app_review_status: str
    max_video_bytes: int = MAX_VIDEO_BYTES

    @classmethod
    def from_environment(cls, environment: Mapping[str, str] | None = None) -> "Settings":
        environment = os.environ if environment is None else environment
        return cls(
            client_key=environment.get("TIKTOK_CLIENT_KEY", "").strip(),
            client_secret=environment.get("TIKTOK_CLIENT_SECRET", "").strip(),
            redirect_uri=environment.get("TIKTOK_REDIRECT_URI", "").strip(),
            session_secret=environment.get("TIKTOK_SESSION_SECRET", "").strip(),
            public_base_url=environment.get("PUBLIC_BASE_URL", "http://127.0.0.1:8000").rstrip("/"),
            draft_transfer_enabled=_enabled(environment, "ENABLE_TIKTOK_DRAFT_TRANSFER"),
            sandbox_review_mode=_enabled(environment, "TIKTOK_SANDBOX_REVIEW_MODE"),
            app_review_status=environment.get("TIKTOK_APP_REVIEW_STATUS", "").strip().lower(),
        )

    @property
    def oauth_configured(self) -> bool:
        return all((self.client_key, self.client_secret, self.redirect_uri, self.session_secret))

    @property
    def transfer_allowed(self) -> bool:
        return self.draft_transfer_enabled and (
            self.sandbox_review_mode or self.app_review_status == "approved"
        )


@dataclass
class CreatorSession:
    oauth_state: str = ""
    access_token: str = ""
    refresh_token: str = ""
    open_id: str = ""
    display_name: str = ""
    avatar_url: str = ""
    scopes: set[str] = field(default_factory=set)
    video_bytes: bytes = b""
    video_name: str = ""
    video_type: str = ""


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, CreatorSession] = {}

    def get_or_create(self, session_id: str | None = None) -> tuple[str, CreatorSession]:
        if session_id and session_id in self._sessions:
            return session_id, self._sessions[session_id]
        session_id = secrets.token_urlsafe(32)
        session = CreatorSession()
        self._sessions[session_id] = session
        return session_id, session

    def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


class TikTokClient:
    def __init__(self, settings: Settings, opener: Any = urllib.request.urlopen) -> None:
        self.settings = settings
        self.opener = opener

    def authorization_url(self, state: str) -> str:
        query = urllib.parse.urlencode(
            {
                "client_key": self.settings.client_key,
                "scope": "user.info.basic,video.upload",
                "response_type": "code",
                "redirect_uri": self.settings.redirect_uri,
                "state": state,
            }
        )
        return f"{AUTH_URL}?{query}"

    def exchange_code(self, code: str) -> dict[str, Any]:
        body = urllib.parse.urlencode(
            {
                "client_key": self.settings.client_key,
                "client_secret": self.settings.client_secret,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": self.settings.redirect_uri,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            TOKEN_URL,
            data=body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        return self._json(request)

    def user_info(self, access_token: str) -> dict[str, Any]:
        query = urllib.parse.urlencode(
            {"fields": "open_id,display_name,avatar_url"}
        )
        request = urllib.request.Request(
            f"{USER_INFO_URL}?{query}",
            headers={"Authorization": f"Bearer {access_token}"},
            method="GET",
        )
        payload = self._json(request)
        return payload.get("data", {}).get("user", {})

    def send_to_inbox(self, access_token: str, video: bytes, content_type: str) -> str:
        size = len(video)
        init_request = urllib.request.Request(
            INBOX_INIT_URL,
            data=json.dumps(
                {
                    "source_info": {
                        "source": "FILE_UPLOAD",
                        "video_size": size,
                        "chunk_size": size,
                        "total_chunk_count": 1,
                    }
                }
            ).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json; charset=UTF-8",
            },
            method="POST",
        )
        init_payload = self._json(init_request)
        data = init_payload.get("data", {})
        upload_url = data.get("upload_url", "")
        publish_id = data.get("publish_id", "")
        if not upload_url or not publish_id:
            raise RuntimeError("TikTok did not return an upload URL and publish ID")

        upload_request = urllib.request.Request(
            upload_url,
            data=video,
            headers={
                "Content-Type": content_type,
                "Content-Length": str(size),
                "Content-Range": f"bytes 0-{size - 1}/{size}",
            },
            method="PUT",
        )
        with self.opener(upload_request, timeout=60) as response:
            response.read(65536)
        return publish_id

    def _json(self, request: urllib.request.Request) -> dict[str, Any]:
        with self.opener(request, timeout=20) as response:
            payload = json.loads(response.read(65536).decode("utf-8"))
        error = payload.get("error", {})
        if error and error.get("code") not in {None, "", "ok"}:
            raise RuntimeError("TikTok rejected the request")
        return payload


def _signed_state(session_id: str, settings: Settings) -> str:
    nonce = secrets.token_urlsafe(18)
    issued_at = str(int(time.time()))
    body = f"{session_id}.{issued_at}.{nonce}"
    signature = hmac.new(
        settings.session_secret.encode("utf-8"), body.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    return f"{body}.{signature}"


def _valid_state(state: str, session_id: str, settings: Settings) -> bool:
    try:
        state_session, issued_at, _nonce, signature = state.rsplit(".", 3)
        age = int(time.time()) - int(issued_at)
    except (ValueError, TypeError):
        return False
    body = state.rsplit(".", 1)[0]
    expected = hmac.new(
        settings.session_secret.encode("utf-8"), body.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    return state_session == session_id and 0 <= age <= 600 and hmac.compare_digest(signature, expected)


def _page(content: str, *, title: str = "Creadores | La Estratosférica") -> HTMLResponse:
    return HTMLResponse(
        """<!doctype html><html lang=\"es\"><head><meta charset=\"utf-8\">
<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
<title>""" + html.escape(title) + """</title><style>
:root{--bg:#090313;--panel:#170a2b;--purple:#7628ff;--cyan:#25e9ff;--pink:#ff2dbb;--text:#fff;--muted:#bcb2ce}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 80% 0,#30105b 0,transparent 34%),var(--bg);color:var(--text);font-family:Inter,system-ui,sans-serif;min-height:100vh}
main{width:min(880px,92vw);margin:auto;padding:42px 0 72px}.brand{font-weight:900;letter-spacing:.08em}.eyebrow{color:var(--cyan);font-size:.8rem;text-transform:uppercase;letter-spacing:.16em;margin-top:80px}h1{font-size:clamp(2.4rem,7vw,5rem);line-height:.95;margin:.35em 0}.lead{font-size:1.15rem;line-height:1.6;color:var(--muted);max-width:650px}.card{background:linear-gradient(145deg,rgba(38,15,69,.96),rgba(18,7,35,.96));border:1px solid #54279a;border-radius:24px;padding:26px;margin:28px 0;box-shadow:0 25px 80px #0008}.steps{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.step{padding:18px;border-radius:16px;background:#ffffff0a;border:1px solid #ffffff14}.num{display:block;color:var(--pink);font-weight:900;font-size:1.5rem}.button,button{display:inline-block;border:0;border-radius:999px;background:var(--purple);color:#fff;padding:15px 24px;font-weight:800;text-decoration:none;cursor:pointer}.button:hover,button:hover{background:#8d49ff}.secondary{background:#ffffff12;border:1px solid #ffffff30}.status{padding:12px 16px;border-radius:12px;background:#25e9ff18;color:var(--cyan);margin:16px 0}.warning{background:#ffb0201c;color:#ffd089}.profile{display:flex;align-items:center;gap:14px}.profile img{width:58px;height:58px;border-radius:50%;object-fit:cover}.drop{display:block;border:2px dashed #7040a7;border-radius:18px;padding:28px;text-align:center;margin:18px 0}.file{color:var(--muted);font-size:.9rem}.fine{font-size:.82rem;color:var(--muted);line-height:1.5}@media(max-width:650px){.steps{grid-template-columns:1fr}.eyebrow{margin-top:42px}}
</style></head><body><main><div class=\"brand\">LA ESTRATOSFÉRICA</div>""" + content + """</main></body></html>"""
    )


def create_app(
    settings: Settings | None = None,
    *,
    store: SessionStore | None = None,
    client: TikTokClient | None = None,
) -> FastAPI:
    settings = settings or Settings.from_environment()
    store = store or SessionStore()
    client = client or TikTokClient(settings)
    app = FastAPI(title="La Estratosférica Creator Portal", docs_url=None, redoc_url=None)

    def session_for(request: Request) -> tuple[str, CreatorSession]:
        return store.get_or_create(request.cookies.get(SESSION_COOKIE))

    def attach_cookie(response: Response, session_id: str) -> Response:
        response.set_cookie(
            SESSION_COOKIE,
            session_id,
            max_age=3600,
            httponly=True,
            secure=settings.public_base_url.startswith("https://"),
            samesite="lax",
        )
        return response

    @app.get("/", response_class=HTMLResponse)
    def home(request: Request) -> Response:
        session_id, session = session_for(request)
        if session.access_token:
            response = RedirectResponse("/creator", status_code=303)
            return attach_cookie(response, session_id)
        configured = settings.oauth_configured
        status = (
            "Integración sandbox lista para conectar."
            if configured
            else "Demo segura: faltan variables OAuth del entorno."
        )
        connect = (
            '<a class="button" href="/oauth/tiktok/start">Conectar TikTok</a>'
            if configured
            else '<span class="button secondary">Conexión aún bloqueada</span>'
        )
        response = _page(
            f"""<div class=\"eyebrow\">Portal para creadores</div>
<h1>Tu contenido.<br>Tu cuenta.<br>Tu decisión.</h1>
<p class=\"lead\">Conecta tu cuenta y envía un video propio a tus borradores de TikTok. Tú lo revisas, editas y decides si publicarlo dentro de TikTok.</p>
<div class=\"card\"><div class=\"steps\"><div class=\"step\"><span class=\"num\">01</span>Conecta tu cuenta</div><div class=\"step\"><span class=\"num\">02</span>Elige tu video</div><div class=\"step\"><span class=\"num\">03</span>Recíbelo como borrador</div></div><div class=\"status\">{html.escape(status)}</div>{connect}<p class=\"fine\">Solicitamos únicamente identidad básica y envío a borradores. La Estratosférica nunca publica por ti.</p></div>"""
        )
        return attach_cookie(response, session_id)

    @app.get("/oauth/tiktok/start")
    def oauth_start(request: Request) -> Response:
        if not settings.oauth_configured:
            raise HTTPException(503, "TikTok OAuth is not configured")
        session_id, session = session_for(request)
        session.oauth_state = _signed_state(session_id, settings)
        response = RedirectResponse(client.authorization_url(session.oauth_state), status_code=303)
        return attach_cookie(response, session_id)

    @app.get("/oauth/tiktok/callback")
    def oauth_callback(request: Request, code: str = "", state: str = "", error: str = "") -> Response:
        session_id, session = session_for(request)
        if error:
            raise HTTPException(400, "TikTok authorization was cancelled")
        if not code or not state or state != session.oauth_state or not _valid_state(state, session_id, settings):
            raise HTTPException(400, "Invalid or expired OAuth state")
        token = client.exchange_code(code)
        session.access_token = token.get("access_token", "")
        session.refresh_token = token.get("refresh_token", "")
        session.open_id = token.get("open_id", "")
        session.scopes = {item.strip() for item in token.get("scope", "").split(",") if item.strip()}
        user = client.user_info(session.access_token)
        session.display_name = user.get("display_name", "Creator")
        session.avatar_url = user.get("avatar_url", "")
        session.oauth_state = ""
        response = RedirectResponse("/creator", status_code=303)
        return attach_cookie(response, session_id)

    @app.get("/creator", response_class=HTMLResponse)
    def creator(request: Request) -> Response:
        session_id, session = session_for(request)
        if not session.access_token:
            response = RedirectResponse("/", status_code=303)
            return attach_cookie(response, session_id)
        avatar = (
            f'<img src="{html.escape(session.avatar_url, quote=True)}" alt="Avatar">'
            if session.avatar_url
            else ""
        )
        transfer_state = (
            "Sandbox autorizado para transferir un único borrador."
            if settings.transfer_allowed
            else "Transferencia bloqueada hasta una prueba sandbox expresamente autorizada."
        )
        response = _page(
            f"""<div class=\"eyebrow\">Cuenta conectada</div><div class=\"card\"><div class=\"profile\">{avatar}<div><strong>{html.escape(session.display_name)}</strong><div class=\"fine\">TikTok conectado · publicación bajo tu control</div></div></div></div>
<div class=\"card\"><h2>Envía tu video a borradores</h2><p class=\"lead\">Selecciona un MP4 o MOV propio. Máximo 50 MB para esta prueba.</p><label class=\"drop\"><input id=\"video\" type=\"file\" accept=\"video/mp4,video/quicktime\"><span id=\"fileText\" class=\"file\">Seleccionar video</span></label><div id=\"message\" class=\"status warning\">{html.escape(transfer_state)}</div><button id=\"send\" type=\"button\">Enviar como borrador</button> <a class=\"button secondary\" href=\"/disconnect\">Desconectar</a></div>
<script>const input=document.getElementById('video'),message=document.getElementById('message');input.onchange=()=>document.getElementById('fileText').textContent=input.files[0]?.name||'Seleccionar video';document.getElementById('send').onclick=async()=>{{const file=input.files[0];if(!file){{message.textContent='Selecciona un video primero.';return}}message.textContent='Preparando transferencia segura…';const response=await fetch('/api/video',{{method:'POST',headers:{{'Content-Type':file.type,'X-File-Name':encodeURIComponent(file.name)}},body:file}});const data=await response.json();message.textContent=data.message||data.detail||'Respuesta recibida';}};</script>"""
        )
        return attach_cookie(response, session_id)

    @app.post("/api/video")
    async def upload_video(request: Request) -> dict[str, Any]:
        _session_id, session = session_for(request)
        if not session.access_token:
            raise HTTPException(401, "Connect TikTok first")
        content_type = request.headers.get("content-type", "").split(";", 1)[0].lower()
        if content_type not in ALLOWED_VIDEO_TYPES:
            raise HTTPException(415, "Use an MP4 or MOV video")
        video = await request.body()
        if not video or len(video) > settings.max_video_bytes:
            raise HTTPException(413, "Video must be between 1 byte and 50 MB")
        if "video.upload" not in session.scopes:
            raise HTTPException(403, "The connected account did not grant video.upload")
        session.video_bytes = video
        session.video_name = urllib.parse.unquote(request.headers.get("x-file-name", "video.mp4"))[:180]
        session.video_type = content_type
        if not settings.transfer_allowed:
            return {
                "status": "held_for_review",
                "message": "Video validado. La transferencia permanece bloqueada hasta autorizar la prueba sandbox.",
                "external_write_attempted": False,
            }
        publish_id = client.send_to_inbox(session.access_token, video, content_type)
        session.video_bytes = b""
        return {
            "status": "sent_to_tiktok_inbox",
            "message": "Borrador transferido. Ábrelo en TikTok para revisarlo y decidir si publicarlo.",
            "receipt": publish_id,
            "published": False,
        }

    @app.get("/disconnect")
    def disconnect(request: Request) -> Response:
        session_id = request.cookies.get(SESSION_COOKIE, "")
        if session_id:
            store.delete(session_id)
        response = RedirectResponse("/", status_code=303)
        response.delete_cookie(SESSION_COOKIE)
        return response

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "oauth_configured": settings.oauth_configured,
            "draft_transfer_enabled": settings.draft_transfer_enabled,
            "sandbox_review_mode": settings.sandbox_review_mode,
            "transfer_allowed": settings.transfer_allowed,
            "direct_post_enabled": False,
        }

    return app


app = create_app()
