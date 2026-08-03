const GRAPH_VERSION = "v25.0";
const HANDOFF_KEY = "facebook-page-token-pending.json";
const STATE_MESSAGE = "laestratosferica/facebook-page-token-rotation/v1";

function text(status, message) {
  return new Response(message, {
    status,
    headers: { "content-type": "text/plain; charset=utf-8", "cache-control": "no-store" },
  });
}

function base64Url(bytes) {
  let binary = "";
  for (const byte of new Uint8Array(bytes)) binary += String.fromCharCode(byte);
  return btoa(binary).replaceAll("+", "-").replaceAll("/", "_").replace(/=+$/u, "");
}

async function oauthState(appSecret) {
  const key = await crypto.subtle.importKey(
    "raw", new TextEncoder().encode(appSecret), { name: "HMAC", hash: "SHA-256" }, false, ["sign"],
  );
  return base64Url(await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(STATE_MESSAGE)));
}

function sameValue(left, right) {
  if (!left || !right || left.length !== right.length) return false;
  let difference = 0;
  for (let index = 0; index < left.length; index += 1) difference |= left.charCodeAt(index) ^ right.charCodeAt(index);
  return difference === 0;
}

async function graph(path, searchParams) {
  const response = await fetch(`https://graph.facebook.com/${GRAPH_VERSION}/${path}?${searchParams}`);
  const body = await response.json().catch(() => ({}));
  if (!response.ok || body.error) throw new Error("meta_request_failed");
  return body;
}

async function rotatePageToken(code, env, requestUrl) {
  const redirectUri = `${requestUrl.origin}/facebook/callback`;
  const short = await graph("oauth/access_token", new URLSearchParams({
    client_id: env.FB_APP_ID, client_secret: env.FB_APP_SECRET, redirect_uri: redirectUri, code,
  }));
  const longLived = await graph("oauth/access_token", new URLSearchParams({
    grant_type: "fb_exchange_token", client_id: env.FB_APP_ID, client_secret: env.FB_APP_SECRET,
    fb_exchange_token: short.access_token,
  }));
  const accounts = await graph("me/accounts", new URLSearchParams({
    access_token: longLived.access_token, fields: "id,name,access_token",
  }));
  const page = (accounts.data || []).find((candidate) => String(candidate.id) === String(env.FB_PAGE_ID));
  if (!page?.access_token) throw new Error("expected_page_not_authorized");
  const profile = await graph(env.FB_PAGE_ID, new URLSearchParams({
    access_token: page.access_token, fields: "id,name,link",
  }));
  if (String(profile.id) !== String(env.FB_PAGE_ID)) throw new Error("page_identity_mismatch");
  return { token: page.access_token, page: { id: profile.id, name: profile.name, link: profile.link } };
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname === "/health") return text(200, "ok");
    if (url.pathname === "/facebook/callback") {
      if (request.method !== "GET") return text(405, "Método no permitido.");
      const state = url.searchParams.get("state");
      if (!sameValue(state, await oauthState(env.FB_APP_SECRET))) return text(400, "Autorización no válida.");
      const code = url.searchParams.get("code");
      if (!code) return text(400, "Meta no entregó el código de autorización.");
      try {
        const result = await rotatePageToken(code, env, url);
        await env.FB_OAUTH_HANDOFF.put(HANDOFF_KEY, JSON.stringify({ ...result, stored_at: new Date().toISOString() }), {
          httpMetadata: { contentType: "application/json" },
        });
        return text(200, "Autorización recibida. Puedes volver a GitHub; el token se renovará sin mostrarlo.");
      } catch {
        return text(400, "No fue posible verificar la cuenta de Facebook autorizada.");
      }
    }
    if (url.pathname === "/facebook/handoff") {
      const authorization = request.headers.get("authorization") || "";
      if (!sameValue(authorization, `Bearer ${env.FB_APP_SECRET}`)) return text(401, "No autorizado.");
      const object = await env.FB_OAUTH_HANDOFF.get(HANDOFF_KEY);
      if (!object) return text(404, "No hay renovación pendiente.");
      await env.FB_OAUTH_HANDOFF.delete(HANDOFF_KEY);
      return new Response(object.body, { headers: { "content-type": "application/json", "cache-control": "no-store" } });
    }
    return text(404, "No encontrado.");
  },
};
