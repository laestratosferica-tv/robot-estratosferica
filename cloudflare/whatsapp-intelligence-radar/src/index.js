const COMMANDS = new Map([
  ["NOTICIA", "editorial_news"], ["TENDENCIA", "editorial_trend"],
  ["PERSONAJE", "character"], ["EFECTO", "visual_effect"],
  ["ROBOT", "robot_knowledge"], ["HERRAMIENTA", "tool"],
  ["APRENDIZAJE", "learning"], ["IDEA", "idea"],
]);

const CATEGORY_RULES = [
  ["character", ["personaje", "avatar", "mascota", "apariencia", "skin", "diseno de personaje"]],
  ["visual_effect", ["efecto", "transicion", "animacion", "filtro", "vfx", "overlay"]],
  ["robot_knowledge", ["robot", "agente", "automatizacion", "memoria", "conocimiento del proyecto"]],
  ["tool", ["herramienta", "aplicacion", "plataforma", "software", "plugin", "servicio"]],
  ["learning", ["aprendizaje", "tutorial", "curso", "guia", "como hacer", "explica"]],
  ["editorial_news", ["noticia", "anuncio", "anuncia", "lanzamiento", "actualizacion", "confirmado", "ultima hora"]],
  ["editorial_trend", ["tendencia", "viral", "trend", "tiktok", "reels", "shorts"]],
  ["idea", ["idea", "propuesta", "podriamos", "opcion", "concepto", "inspiracion"]],
];

const SIGNAL_RETENTION_SECONDS = 90 * 24 * 60 * 60;

const CATEGORY_DECISIONS = {
  editorial_news: { label: "Noticia", contribution: "Posible dato editorial actual para investigar.", take: "Verificar la fuente original, la fecha y el contexto.", discard: "Afirmaciones sin fuente y publicación directa.", action: "Investigar", priority: "alta" },
  editorial_trend: { label: "Tendencia", contribution: "Señal de formato, conversación o comportamiento que puede estar creciendo.", take: "Analizar el patrón y adaptarlo con una ejecución propia.", discard: "Copiar el clip o asumir que es tendencia sin métricas.", action: "Investigar", priority: "alta" },
  character: { label: "Personaje", contribution: "Referencia creativa para ampliar el universo de personajes.", take: "Guardar rasgos útiles y desarrollar una versión original.", discard: "Imitar identidad, voz o diseño protegido.", action: "Guardar", priority: "media" },
  visual_effect: { label: "Efecto", contribution: "Recurso visual que puede mejorar claridad, impacto o retención.", take: "Probar una versión propia en un entorno controlado.", discard: "Reutilizar archivos sin licencia verificable.", action: "Probar", priority: "media" },
  robot_knowledge: { label: "Robot", contribution: "Conocimiento potencial para mejorar automatización o capacidades del proyecto.", take: "Documentar y validar antes de incorporarlo al robot.", discard: "Integrarlo como conocimiento verdadero sin verificación.", action: "Investigar", priority: "alta" },
  tool: { label: "Herramienta", contribution: "Posible mejora operativa, creativa o tecnológica.", take: "Evaluar seguridad, costo, integración y retorno.", discard: "Conectar cuentas o pagar sin una prueba segura.", action: "Evaluar", priority: "media" },
  learning: { label: "Aprendizaje", contribution: "Método o explicación que puede fortalecer el conocimiento interno.", take: "Contrastar el método y guardar lo comprobable.", discard: "Consejos no demostrados o desactualizados.", action: "Aprender", priority: "media" },
  idea: { label: "Idea", contribution: "Punto de partida para contenido, producto o automatización.", take: "Convertirla en una hipótesis pequeña y comprobable.", discard: "Ejecutarla completa sin validar utilidad ni costo.", action: "Diseñar prueba", priority: "media" },
  unclassified: { label: "Sin clasificar", contribution: "Todavía no hay contexto suficiente para determinar su valor.", take: "Revisar manualmente el contenido y su fuente.", discard: "Tomar decisiones o publicar con información incompleta.", action: "Revisar", priority: "baja" },
};

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
  const links = [...trimmed.matchAll(/https?:\/\/[^\s]+/gu)].map((match) => match[0]).slice(0, 5);
  const submittedCategory = COMMANDS.get(first.replace(/:$/u, "").toUpperCase());
  if (submittedCategory) return { category: submittedCategory, text: trimmed.slice(0, 4000), links, confidence: "high", classification_method: "submitted_label" };

  const normalized = trimmed.toLowerCase().normalize("NFD").replace(/[\u0300-\u036f]/gu, "");
  const scored = CATEGORY_RULES.map(([category, terms], index) => ({
    category,
    index,
    score: terms.reduce((total, term) => total + (normalized.includes(term) ? 1 : 0), 0),
  })).sort((left, right) => right.score - left.score || left.index - right.index);
  let category = scored[0].score > 0 ? scored[0].category : "unclassified";
  let method = scored[0].score > 0 ? "natural_language_rules" : "needs_triage";

  if ((category === "unclassified" || (links.length === 1 && trimmed === links[0])) && links.some((link) => {
    try { return /(?:^|\.)tiktok\.com$/iu.test(new URL(link).hostname); } catch { return false; }
  })) {
    category = "editorial_trend";
    method = "source_domain";
  }
  return {
    category,
    text: trimmed.slice(0, 4000),
    links,
    confidence: category === "unclassified" ? "low" : "medium",
    classification_method: method,
  };
}

function assess(entry) {
  const rule = CATEGORY_DECISIONS[entry.category] || CATEGORY_DECISIONS.unclassified;
  return {
    category_label: rule.label,
    contribution: rule.contribution,
    take: rule.take,
    discard: rule.discard,
    action: rule.action,
    priority: entry.confidence === "low" ? "baja" : rule.priority,
    verification: entry.links?.length ? "Fuente y derechos pendientes de verificar." : "Contexto y evidencia pendientes de verificar.",
    decision_status: entry.category === "unclassified" ? "needs_review" : "candidate",
  };
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
    const assessment = assess(entry);
    const id = `radar:received:${message.id}`;
    await env.RADAR_KV.put(id, JSON.stringify({
      schema: "estratosferica_intelligence_signal_v1", id, status: "received",
      source: "whatsapp", received_at: new Date().toISOString(), sender_hash: await digest(message.from || ""),
      source_message_id: message.id, rights_status: "not_verified", editorial_status: "not_eligible",
      knowledge_status: "not_verified", recommended_action: "triage", ...entry,
      assessment,
    }), { expirationTtl: SIGNAL_RETENTION_SECONDS });
  }
  return response(200, "EVENT_RECEIVED");
}

function authorized(request, env) {
  const authorization = request.headers.get("authorization") || "";
  if (authorization.startsWith("Bearer ")) return sameValue(authorization.slice(7), env.RADAR_ADMIN_TOKEN || "");
  if (!authorization.startsWith("Basic ")) return false;
  try {
    const decoded = atob(authorization.slice(6));
    const separator = decoded.indexOf(":");
    return separator >= 0 && sameValue(decoded.slice(separator + 1), env.RADAR_ADMIN_TOKEN || "");
  } catch { return false; }
}

async function loadRecent(env, limit = 50) {
  if (!env.RADAR_KV) return [];
  const listed = await env.RADAR_KV.list({ prefix: "radar:received:", limit });
  const records = await Promise.all((listed.keys || []).map(async ({ name }) => env.RADAR_KV.get(name, "json")));
  return records.filter(Boolean).sort((a, b) => String(b.received_at).localeCompare(String(a.received_at)));
}

async function recentSignals(request, env) {
  if (!authorized(request, env)) return response(401, "No autorizado.");
  if (!env.RADAR_KV) return response(503, "Almacenamiento no disponible.");
  const records = await loadRecent(env, 20);
  const signals = records.map((record) => ({ id: record.id, received_at: record.received_at, category: record.category, text: record.text, links: record.links, status: record.status, assessment: record.assessment || assess(record) }));
  return Response.json({ count: signals.length, signals }, { headers: { "cache-control": "no-store" } });
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/gu, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;" }[character]));
}

async function dashboard(request, env) {
  if (!authorized(request, env)) return new Response("Acceso privado", { status: 401, headers: { "www-authenticate": "Basic realm=\"Radar La Estratosférica\"", "cache-control": "no-store" } });
  const records = await loadRecent(env);
  const cards = records.map((record) => {
    const result = record.assessment || assess(record);
    const link = record.links?.[0] ? `<a href="${escapeHtml(record.links[0])}" target="_blank" rel="noopener noreferrer">Abrir fuente</a>` : "Sin enlace";
    return `<article><div class="top"><span class="tag">${escapeHtml(result.category_label)}</span><span class="priority ${escapeHtml(result.priority)}">${escapeHtml(result.priority)}</span></div><p class="source">${escapeHtml(record.text)}</p><dl><dt>Qué aporta</dt><dd>${escapeHtml(result.contribution)}</dd><dt>Qué tomamos</dt><dd>${escapeHtml(result.take)}</dd><dt>Qué descartamos</dt><dd>${escapeHtml(result.discard)}</dd><dt>Acción</dt><dd>${escapeHtml(result.action)}</dd><dt>Verificación</dt><dd>${escapeHtml(result.verification)}</dd></dl><footer>${link}<time>${escapeHtml(record.received_at)}</time></footer></article>`;
  }).join("");
  const today = records.filter((record) => Date.now() - new Date(record.received_at).getTime() <= 24 * 60 * 60 * 1000);
  const candidates = today.filter((record) => (record.assessment || assess(record)).decision_status === "candidate").length;
  const reviews = today.length - candidates;
  const html = `<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Radar WhatsApp | La Estratosférica</title><style>:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:#080a12;color:#f6f7fb;font:15px/1.5 system-ui,sans-serif}main{max-width:1100px;margin:auto;padding:32px 18px 64px}header{padding:24px 0 18px}h1{font-size:clamp(30px,6vw,54px);margin:0}.lead{color:#adb5d1}.summary{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:22px 0}.summary div,article{background:#111525;border:1px solid #252c48;border-radius:18px;padding:18px}.summary strong{display:block;font-size:30px;color:#79f2c0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:14px}.top,footer{display:flex;justify-content:space-between;gap:12px;align-items:center}.tag{color:#79f2c0;font-weight:750}.priority{font-size:12px;text-transform:uppercase;padding:4px 9px;border-radius:99px;background:#30364b}.alta{background:#7d2636}.media{background:#66521f}.baja{background:#30425c}.source{min-height:48px;color:#dfe3f3;overflow-wrap:anywhere}dl{margin:0}dt{margin-top:12px;font-weight:750;color:#a4acd0}dd{margin:2px 0}footer{border-top:1px solid #252c48;margin-top:18px;padding-top:14px;font-size:12px;color:#8991ae}a{color:#79f2c0}time{text-align:right}@media(max-width:560px){.summary{grid-template-columns:1fr}.grid{grid-template-columns:1fr}}</style></head><body><main><header><div class="tag">LA ESTRATOSFÉRICA</div><h1>Radar WhatsApp</h1><p class="lead">Qué aporta, qué tomamos y qué descartamos. Nada se publica automáticamente.</p></header><section class="summary"><div><strong>${today.length}</strong>recibidos en 24 horas</div><div><strong>${candidates}</strong>candidatos</div><div><strong>${reviews}</strong>por revisar</div></section><h2>Plan del día</h2><p class="lead">Priorizar ${candidates} candidato(s), verificar sus fuentes y revisar ${reviews} señal(es) sin contexto suficiente.</p><section class="grid">${cards || "<article>Aún no hay señales recibidas.</article>"}</section></main></body></html>`;
  return new Response(html, { headers: { "content-type": "text/html; charset=utf-8", "cache-control": "no-store", "x-frame-options": "DENY", "content-security-policy": "default-src 'none'; style-src 'unsafe-inline'; img-src 'self'; base-uri 'none'; form-action 'none'" } });
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname === "/privacy") return new Response(PRIVACY_NOTICE, { headers: { "content-type": "text/html; charset=utf-8", "cache-control": "public, max-age=3600" } });
    if (url.pathname === "/health") return Response.json({ ok: true, whatsapp_radar_enabled: env.ENABLE_WHATSAPP_RADAR === "true", automatic_publication: false, automatic_knowledge_approval: false });
    if (url.pathname === "/dashboard" && request.method === "GET") return dashboard(request, env);
    if (url.pathname === "/internal/recent" && request.method === "GET") return recentSignals(request, env);
    if (url.pathname !== "/webhooks/whatsapp") return response(404, "No encontrado.");
    if (request.method === "GET") {
      if (url.searchParams.get("hub.mode") !== "subscribe" || !sameValue(url.searchParams.get("hub.verify_token") || "", env.WHATSAPP_WEBHOOK_VERIFY_TOKEN || "")) return response(403, "Verificación no válida.");
      return response(200, url.searchParams.get("hub.challenge") || "");
    }
    if (request.method === "POST") return acceptWebhook(request, env);
    return response(405, "Método no permitido.");
  },
};
