const COMMANDS = new Map([
  ["NOTICIA", "editorial_news"], ["TENDENCIA", "editorial_trend"],
  ["PERSONAJE", "character"], ["EFECTO", "visual_effect"],
  ["ROBOT", "robot_knowledge"], ["HERRAMIENTA", "tool"],
  ["APRENDIZAJE", "learning"], ["IDEA", "idea"],
]);

const SIGNAL_RETENTION_SECONDS = 90 * 24 * 60 * 60;

const PRIVACY_NOTICE = `<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Política de privacidad | La Estratosférica</title>
<style>body{font:16px/1.6 system-ui,sans-serif;max-width:760px;margin:48px auto;padding:0 20px;color:#171717}h1,h2{line-height:1.2}small{color:#666}</style></head>
<body><h1>Política de privacidad del Radar WhatsApp</h1><small>Última actualización: 11 de agosto de 2026</small>
<p>La Estratosférica utiliza este servicio para recibir voluntariamente enlaces, ideas y señales sobre videojuegos, tecnología, inteligencia artificial y cultura digital mediante WhatsApp.</p>
<h2>Datos tratados</h2><p>Procesamos el texto y los enlaces enviados, el identificador técnico del mensaje, la fecha de recepción y una huella criptográfica irreversible del remitente. No almacenamos el número telefónico en texto claro.</p>
<h2>Finalidad y decisiones</h2><p>Los datos se usan para clasificar señales como noticia, tendencia, personaje, efecto, robot, herramienta, aprendizaje o idea. Ningún mensaje se publica ni se incorpora automáticamente como conocimiento verificado.</p>
<h2>Proveedores y conservación</h2><p>Meta Platforms proporciona WhatsApp Business y Cloudflare aloja el procesamiento y almacenamiento. Las señales se conservan hasta 90 días y después se eliminan automáticamente, salvo obligación legal o solicitud válida de conservación.</p>
<h2>Derechos y contacto</h2><p>Puedes solicitar acceso, corrección o eliminación de la información asociada a tus envíos escribiendo a <a href="mailto:laestratosferica@gmail.com">laestratosferica@gmail.com</a>. También puedes dejar de enviar información en cualquier momento.</p>
<h2>Seguridad</h2><p>Verificamos la firma de Meta, limitamos la información almacenada y mantenemos separados los contenidos recibidos de los sistemas de publicación.</p>
</body></html>`;

function response(status, body) {
  return new Response(body, { status, headers: { "content-type": "text/plain; charset=utf-8", "cache-control": "no-store" } });
}

function sameValue(left, right) {
  if (!left || !right || left.length !== right.length) return false;
  let difference = 0;
  for (let index = 0; index < left.length; index += 1) difference |= left.charCodeAt(index) ^ right.charCodeAt(index);
  return difference === 0;
}

async function hmacHex(secret, content) {
  const key = await crypto.subtle.importKey("raw", new TextEncoder().encode(secret), { name: "HMAC", hash: "SHA-256" }, false, ["sign"]);
  const signature = await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(content));
  return [...new Uint8Array(signature)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

async function digest(value) {
  const hash = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(value));
  return [...new Uint8Array(hash)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

function classify(text) {
  const trimmed = text.trim();
  const [first = ""] = trimmed.split(/\s+/u);
  const category = COMMANDS.get(first.replace(/:$/u, "").toUpperCase()) || "unclassified";
  const links = [...trimmed.matchAll(/https?:\/\/[^\s]+/gu)].map((match) => match[0]).slice(0, 5);
  return { category, text: trimmed.slice(0, 4000), links, confidence: category === "unclassified" ? "needs_triage" : "submitted_signal" };
}

function messages(payload) {
  const changes = payload?.entry?.flatMap((entry) => entry.changes || []) || [];
  return changes.flatMap((change) => change?.value?.messages || []).filter((message) => message.type === "text" && message.text?.body);
}

async function acceptWebhook(request, env) {
  if (env.ENABLE_WHATSAPP_RADAR !== "true") return response(503, "Radar de WhatsApp aún no está activo.");
  if (!env.WHATSAPP_APP_SECRET || !env.RADAR_KV) return response(503, "Configuración del radar incompleta.");
  const raw = await request.text();
  const received = request.headers.get("x-hub-signature-256") || "";
  const expected = `sha256=${await hmacHex(env.WHATSAPP_APP_SECRET, raw)}`;
  if (!sameValue(received, expected)) return response(401, "Firma no válida.");
  let payload;
  try { payload = JSON.parse(raw); } catch { return response(400, "Evento inválido."); }
  const incoming = messages(payload);
  for (const message of incoming) {
    const entry = classify(message.text.body);
    const id = `radar:received:${message.id}`;
    await env.RADAR_KV.put(id, JSON.stringify({
      schema: "estratosferica_intelligence_signal_v1", id, status: "received",
      source: "whatsapp", received_at: new Date().toISOString(), sender_hash: await digest(message.from || ""),
      source_message_id: message.id, rights_status: "not_verified", editorial_status: "not_eligible",
      knowledge_status: "not_verified", recommended_action: "triage", ...entry,
    }), { expirationTtl: SIGNAL_RETENTION_SECONDS });
  }
  return response(200, "EVENT_RECEIVED");
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname === "/privacy") return new Response(PRIVACY_NOTICE, { headers: { "content-type": "text/html; charset=utf-8", "cache-control": "public, max-age=3600" } });
    if (url.pathname === "/health") return Response.json({ ok: true, whatsapp_radar_enabled: env.ENABLE_WHATSAPP_RADAR === "true", automatic_publication: false, automatic_knowledge_approval: false });
    if (url.pathname !== "/webhooks/whatsapp") return response(404, "No encontrado.");
    if (request.method === "GET") {
      if (url.searchParams.get("hub.mode") !== "subscribe" || !sameValue(url.searchParams.get("hub.verify_token") || "", env.WHATSAPP_WEBHOOK_VERIFY_TOKEN || "")) return response(403, "Verificación no válida.");
      return response(200, url.searchParams.get("hub.challenge") || "");
    }
    if (request.method === "POST") return acceptWebhook(request, env);
    return response(405, "Método no permitido.");
  },
};
