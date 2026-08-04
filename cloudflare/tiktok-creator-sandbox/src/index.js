const AUTH_URL = "https://www.tiktok.com/v2/auth/authorize/";
const TOKEN_URL = "https://open.tiktokapis.com/v2/oauth/token/";
const USER_URL = "https://open.tiktokapis.com/v2/user/info/";
const INBOX_URL = "https://open.tiktokapis.com/v2/post/publish/inbox/video/init/";
const COOKIE = "estratosferica_creator";
const STATE_COOKIE = "estratosferica_oauth_state";
const MAX_VIDEO_BYTES = 50 * 1024 * 1024;

const encoder = new TextEncoder();
const decoder = new TextDecoder();

function base64Url(bytes) {
  let binary = "";
  for (const byte of new Uint8Array(bytes)) binary += String.fromCharCode(byte);
  return btoa(binary).replaceAll("+", "-").replaceAll("/", "_").replace(/=+$/u, "");
}

function fromBase64Url(value) {
  const padded = value.replaceAll("-", "+").replaceAll("_", "/").padEnd(Math.ceil(value.length / 4) * 4, "=");
  const binary = atob(padded);
  return Uint8Array.from(binary, (character) => character.charCodeAt(0));
}

function cookie(request, name) {
  const match = (request.headers.get("cookie") || "").split(/;\s*/u).find((item) => item.startsWith(`${name}=`));
  return match ? match.slice(name.length + 1) : "";
}

function setCookie(name, value, maxAge = 3600) {
  return `${name}=${value}; Max-Age=${maxAge}; Path=/; HttpOnly; Secure; SameSite=Lax`;
}

async function aesKey(secret) {
  const digest = await crypto.subtle.digest("SHA-256", encoder.encode(secret));
  return crypto.subtle.importKey("raw", digest, "AES-GCM", false, ["encrypt", "decrypt"]);
}

async function seal(payload, secret) {
  const iv = crypto.getRandomValues(new Uint8Array(12));
  const encrypted = await crypto.subtle.encrypt({ name: "AES-GCM", iv }, await aesKey(secret), encoder.encode(JSON.stringify(payload)));
  return `${base64Url(iv)}.${base64Url(encrypted)}`;
}

async function unseal(value, secret) {
  try {
    const [iv, encrypted] = value.split(".").map(fromBase64Url);
    const clear = await crypto.subtle.decrypt({ name: "AES-GCM", iv }, await aesKey(secret), encrypted);
    return JSON.parse(decoder.decode(clear));
  } catch {
    return null;
  }
}

function allowed(env) {
  return env.ENABLE_TIKTOK_DRAFT_TRANSFER === "true" &&
    (env.TIKTOK_SANDBOX_REVIEW_MODE === "true" || env.TIKTOK_APP_REVIEW_STATUS === "approved");
}

function page(content) {
  return new Response(`<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Creadores | La Estratosférica</title><style>
:root{--bg:#090313;--panel:#170a2b;--purple:#7628ff;--cyan:#25e9ff;--pink:#ff2dbb;--text:#fff;--muted:#bcb2ce}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 80% 0,#30105b 0,transparent 34%),var(--bg);color:var(--text);font-family:Inter,system-ui,sans-serif;min-height:100vh}main{width:min(880px,92vw);margin:auto;padding:42px 0 72px}.brand{font-weight:900;letter-spacing:.08em}.eyebrow{color:var(--cyan);font-size:.8rem;text-transform:uppercase;letter-spacing:.16em;margin-top:80px}h1{font-size:clamp(2.4rem,7vw,5rem);line-height:.95;margin:.35em 0}.lead{font-size:1.15rem;line-height:1.6;color:var(--muted);max-width:650px}.card{background:linear-gradient(145deg,rgba(38,15,69,.96),rgba(18,7,35,.96));border:1px solid #54279a;border-radius:24px;padding:26px;margin:28px 0}.steps{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.step{padding:18px;border-radius:16px;background:#ffffff0a;border:1px solid #ffffff14}.num{display:block;color:var(--pink);font-weight:900;font-size:1.5rem}.button,button{display:inline-block;border:0;border-radius:999px;background:var(--purple);color:#fff;padding:15px 24px;font-weight:800;text-decoration:none;cursor:pointer}.secondary{background:#ffffff12;border:1px solid #ffffff30}.status{padding:12px 16px;border-radius:12px;background:#25e9ff18;color:var(--cyan);margin:16px 0}.profile{display:flex;align-items:center;gap:14px}.profile img{width:58px;height:58px;border-radius:50%}.drop{display:block;border:2px dashed #7040a7;border-radius:18px;padding:28px;text-align:center;margin:18px 0}.fine{font-size:.82rem;color:var(--muted);line-height:1.5}@media(max-width:650px){.steps{grid-template-columns:1fr}.eyebrow{margin-top:42px}}</style></head><body><main><div class="brand">LA ESTRATOSFÉRICA</div>${content}</main></body></html>`, { headers: { "content-type": "text/html; charset=utf-8", "cache-control": "no-store" } });
}

function home(configured) {
  const action = configured ? '<a class="button" href="/oauth/tiktok/start">Conectar TikTok</a>' : '<span class="button secondary">Conexión aún bloqueada</span>';
  return page(`<div class="eyebrow">Portal para creadores</div><h1>Tu contenido.<br>Tu cuenta.<br>Tu decisión.</h1><p class="lead">Conecta tu cuenta y envía un video propio a tus borradores de TikTok. Tú lo revisas, editas y decides si publicarlo dentro de TikTok.</p><div class="card"><div class="steps"><div class="step"><span class="num">01</span>Conecta tu cuenta</div><div class="step"><span class="num">02</span>Elige tu video</div><div class="step"><span class="num">03</span>Recíbelo como borrador</div></div><div class="status">${configured ? "Sandbox listo para conectar." : "Demo segura: faltan variables OAuth."}</div>${action}<p class="fine">Solicitamos únicamente identidad básica y envío a borradores. La Estratosférica nunca publica por ti.</p></div>`);
}

function creator(session, transferAllowed) {
  const avatar = session.avatar_url ? `<img src="${session.avatar_url.replaceAll('"', "")}" alt="Avatar">` : "";
  return page(`<div class="eyebrow">Cuenta conectada</div><div class="card"><div class="profile">${avatar}<div><strong>${session.display_name || "Creator"}</strong><div class="fine">TikTok conectado · publicación bajo tu control</div></div></div></div><div class="card"><h2>Envía tu video a borradores</h2><p class="lead">Selecciona un MP4 propio. Máximo 50 MB para esta prueba.</p><label class="drop"><input id="video" type="file" accept="video/mp4"><span id="fileText">Seleccionar video</span></label><div id="message" class="status">${transferAllowed ? "Prueba sandbox autorizada." : "Transferencia bloqueada hasta autorizar la prueba sandbox."}</div><button id="send">Enviar como borrador</button> <a class="button secondary" href="/disconnect">Desconectar</a></div><script>const i=document.querySelector('#video'),m=document.querySelector('#message');i.onchange=()=>document.querySelector('#fileText').textContent=i.files[0]?.name||'Seleccionar video';document.querySelector('#send').onclick=async()=>{const f=i.files[0];if(!f){m.textContent='Selecciona un video.';return}m.textContent='Preparando…';const r=await fetch('/api/video',{method:'POST',headers:{'content-type':f.type},body:f});const d=await r.json();m.textContent=d.message||d.detail||'Respuesta recibida';};</script>`);
}

async function exchange(code, env, redirectUri) {
  const response = await fetch(TOKEN_URL, { method: "POST", headers: { "content-type": "application/x-www-form-urlencoded" }, body: new URLSearchParams({ client_key: env.TIKTOK_CLIENT_KEY, client_secret: env.TIKTOK_CLIENT_SECRET, code, grant_type: "authorization_code", redirect_uri: redirectUri }) });
  if (!response.ok) throw new Error("token_exchange_failed");
  return response.json();
}

async function profile(accessToken) {
  const response = await fetch(`${USER_URL}?fields=open_id,display_name,avatar_url`, { headers: { authorization: `Bearer ${accessToken}` } });
  if (!response.ok) throw new Error("profile_failed");
  return (await response.json()).data?.user || {};
}

async function sendDraft(video, accessToken) {
  const size = video.byteLength;
  const init = await fetch(INBOX_URL, { method: "POST", headers: { authorization: `Bearer ${accessToken}`, "content-type": "application/json; charset=UTF-8" }, body: JSON.stringify({ source_info: { source: "FILE_UPLOAD", video_size: size, chunk_size: size, total_chunk_count: 1 } }) });
  const payload = await init.json();
  if (!init.ok || !payload.data?.upload_url) throw new Error("draft_init_failed");
  const upload = await fetch(payload.data.upload_url, { method: "PUT", headers: { "content-type": "video/mp4", "content-length": String(size), "content-range": `bytes 0-${size - 1}/${size}` }, body: video });
  if (!upload.ok) throw new Error("draft_upload_failed");
  return payload.data.publish_id;
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const redirectUri = `${url.origin}/oauth/tiktok/callback`;
    const configured = Boolean(env.TIKTOK_CLIENT_KEY && env.TIKTOK_CLIENT_SECRET && env.SESSION_SECRET);
    if (url.pathname === "/health") return Response.json({ ok: true, oauth_configured: configured, draft_transfer_enabled: env.ENABLE_TIKTOK_DRAFT_TRANSFER === "true", sandbox_review_mode: env.TIKTOK_SANDBOX_REVIEW_MODE === "true", transfer_allowed: allowed(env), direct_post_enabled: false });
    if (url.pathname === "/") return home(configured);
    if (url.pathname === "/oauth/tiktok/start") {
      if (!configured) return new Response("OAuth not configured", { status: 503 });
      const state = crypto.randomUUID();
      const target = new URL(AUTH_URL);
      target.search = new URLSearchParams({ client_key: env.TIKTOK_CLIENT_KEY, scope: "user.info.basic,video.upload", response_type: "code", redirect_uri: redirectUri, state });
      return new Response(null, { status: 303, headers: { location: target.toString(), "set-cookie": setCookie(STATE_COOKIE, state, 600) } });
    }
    if (url.pathname === "/oauth/tiktok/callback") {
      if (url.searchParams.get("state") !== cookie(request, STATE_COOKIE)) return new Response("Invalid OAuth state", { status: 400 });
      try {
        const token = await exchange(url.searchParams.get("code") || "", env, redirectUri);
        const user = await profile(token.access_token);
        const session = await seal({ access_token: token.access_token, refresh_token: token.refresh_token, scopes: token.scope, ...user }, env.SESSION_SECRET);
        return new Response(null, { status: 303, headers: { location: "/creator", "set-cookie": setCookie(COOKIE, session) } });
      } catch { return new Response("TikTok authorization failed", { status: 400 }); }
    }
    const session = configured ? await unseal(cookie(request, COOKIE), env.SESSION_SECRET) : null;
    if (url.pathname === "/creator") return session ? creator(session, allowed(env)) : Response.redirect(`${url.origin}/`, 303);
    if (url.pathname === "/api/video" && request.method === "POST") {
      if (!session) return Response.json({ detail: "Conecta TikTok primero." }, { status: 401 });
      if (request.headers.get("content-type") !== "video/mp4") return Response.json({ detail: "Usa un video MP4." }, { status: 415 });
      const video = await request.arrayBuffer();
      if (!video.byteLength || video.byteLength > MAX_VIDEO_BYTES) return Response.json({ detail: "El video debe pesar máximo 50 MB." }, { status: 413 });
      if (!(session.scopes || "").split(",").includes("video.upload")) return Response.json({ detail: "Falta el permiso video.upload." }, { status: 403 });
      if (!allowed(env)) return Response.json({ status: "held", message: "Video validado. La transferencia permanece bloqueada.", external_write_attempted: false });
      try { const receipt = await sendDraft(video, session.access_token); return Response.json({ status: "sent", message: "Borrador transferido. Revísalo dentro de TikTok.", receipt, published: false }); }
      catch { return Response.json({ detail: "TikTok no aceptó la transferencia." }, { status: 502 }); }
    }
    if (url.pathname === "/disconnect") return new Response(null, { status: 303, headers: { location: "/", "set-cookie": setCookie(COOKIE, "", 0) } });
    return new Response("Not found", { status: 404 });
  },
};
