import assert from "node:assert/strict";
import test from "node:test";
import worker from "../src/index.js";

const secret = "test-app-secret";
async function signature(body) {
  const key = await crypto.subtle.importKey("raw", new TextEncoder().encode(secret), { name: "HMAC", hash: "SHA-256" }, false, ["sign"]);
  const bytes = new Uint8Array(await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(body)));
  return `sha256=${[...bytes].map((byte) => byte.toString(16).padStart(2, "0")).join("")}`;
}

test("is inert and cannot publish by default", async () => {
  const response = await worker.fetch(new Request("https://example.com/health"), {});
  assert.deepEqual(await response.json(), { ok: true, whatsapp_radar_enabled: false, automatic_publication: false, automatic_knowledge_approval: false });
});

test("verifies Meta's handshake without revealing secrets", async () => {
  const ok = await worker.fetch(new Request("https://example.com/webhooks/whatsapp?hub.mode=subscribe&hub.verify_token=verify&hub.challenge=challenge"), { WHATSAPP_WEBHOOK_VERIFY_TOKEN: "verify" });
  assert.equal(ok.status, 200); assert.equal(await ok.text(), "challenge");
  const rejected = await worker.fetch(new Request("https://example.com/webhooks/whatsapp?hub.mode=subscribe&hub.verify_token=wrong"), { WHATSAPP_WEBHOOK_VERIFY_TOKEN: "verify" });
  assert.equal(rejected.status, 403);
});

test("publishes a factual privacy notice", async () => {
  const response = await worker.fetch(new Request("https://example.com/privacy"), {});
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type"), /text\/html/);
  const html = await response.text();
  assert.match(html, /90 días/);
  assert.match(html, /laestratosferica@gmail.com/);
  assert.match(html, /No almacenamos el número telefónico en texto claro/);
});

test("stores a structured knowledge signal only after a signed webhook", async () => {
  const records = new Map();
  const body = JSON.stringify({ entry: [{ changes: [{ value: { messages: [{ id: "wamid.1", from: "573001112233", type: "text", text: { body: "PERSONAJE https://tiktok.com/@demo/video/1 Robot archivista con casco de neón" } }] } }] }] });
  let retention;
  const env = { ENABLE_WHATSAPP_RADAR: "true", WHATSAPP_APP_SECRET: secret, RADAR_KV: { put: async (key, value, options) => { records.set(key, JSON.parse(value)); retention = options; } } };
  const response = await worker.fetch(new Request("https://example.com/webhooks/whatsapp", { method: "POST", headers: { "x-hub-signature-256": await signature(body) }, body }), env);
  assert.equal(response.status, 200); assert.equal(records.size, 1);
  const item = records.get("radar:received:wamid.1");
  assert.equal(item.category, "character"); assert.equal(item.status, "received");
  assert.equal(item.rights_status, "not_verified"); assert.equal(item.editorial_status, "not_eligible");
  assert.notEqual(item.sender_hash, "573001112233");
  assert.deepEqual(retention, { expirationTtl: 7776000 });
});

test("classifies a bare TikTok share automatically without requiring a title", async () => {
  const records = new Map();
  const body = JSON.stringify({ entry: [{ changes: [{ value: { messages: [{ id: "wamid.tiktok", from: "573001112233", type: "text", text: { body: "https://www.tiktok.com/@demo/video/123" } }] } }] }] });
  const env = { ENABLE_WHATSAPP_RADAR: "true", WHATSAPP_APP_SECRET: secret, RADAR_KV: { put: async (key, value) => records.set(key, JSON.parse(value)) } };
  const response = await worker.fetch(new Request("https://example.com/webhooks/whatsapp", { method: "POST", headers: { "x-hub-signature-256": await signature(body) }, body }), env);
  assert.equal(response.status, 200);
  const item = records.get("radar:received:wamid.tiktok");
  assert.equal(item.category, "editorial_trend");
  assert.equal(item.classification_method, "source_domain");
  assert.equal(item.confidence, "medium");
  assert.equal(item.assessment.action, "Investigar");
  assert.match(item.assessment.contribution, /tendencia|Señal/iu);
});

test("classifies natural Spanish messages without command prefixes", async () => {
  const records = new Map();
  const body = JSON.stringify({ entry: [{ changes: [{ value: { messages: [{ id: "wamid.character", from: "573001112233", type: "text", text: { body: "Nueva opción para crear personajes y probar su apariencia" } }] } }] }] });
  const env = { ENABLE_WHATSAPP_RADAR: "true", WHATSAPP_APP_SECRET: secret, RADAR_KV: { put: async (key, value) => records.set(key, JSON.parse(value)) } };
  await worker.fetch(new Request("https://example.com/webhooks/whatsapp", { method: "POST", headers: { "x-hub-signature-256": await signature(body) }, body }), env);
  const item = records.get("radar:received:wamid.character");
  assert.equal(item.category, "character");
  assert.equal(item.classification_method, "natural_language_rules");
});

test("rejects unsigned messages", async () => {
  const response = await worker.fetch(new Request("https://example.com/webhooks/whatsapp", { method: "POST", body: "{}" }), { ENABLE_WHATSAPP_RADAR: "true", WHATSAPP_APP_SECRET: secret, RADAR_KV: { put: async () => assert.fail("must not write") } });
  assert.equal(response.status, 401);
});

test("protects and returns recent signals without sender identifiers", async () => {
  const record = { id: "radar:received:wamid.1", received_at: "2026-08-11T16:54:00.000Z", category: "idea", text: "IDEA prueba del Radar", links: [], status: "received", sender_hash: "hidden" };
  const env = { RADAR_ADMIN_TOKEN: "admin", RADAR_KV: { list: async () => ({ keys: [{ name: record.id }] }), get: async () => record } };
  const denied = await worker.fetch(new Request("https://example.com/internal/recent"), env);
  assert.equal(denied.status, 401);
  const allowed = await worker.fetch(new Request("https://example.com/internal/recent", { headers: { authorization: "Bearer admin" } }), env);
  assert.equal(allowed.status, 200);
  const result = await allowed.json();
  assert.equal(result.count, 1);
  assert.equal(result.signals[0].text, "IDEA prueba del Radar");
  assert.equal(result.signals[0].assessment.action, "Diseñar prueba");
  assert.equal("sender_hash" in result.signals[0], false);
});

test("renders a private results dashboard with decisions and a daily plan", async () => {
  const record = { id: "radar:received:wamid.2", received_at: new Date().toISOString(), category: "tool", confidence: "medium", text: "Mira esta herramienta", links: ["https://example.com/tool"], status: "received" };
  const env = { RADAR_ADMIN_TOKEN: "admin", RADAR_KV: { list: async () => ({ keys: [{ name: record.id }] }), get: async () => record } };
  const denied = await worker.fetch(new Request("https://example.com/dashboard"), env);
  assert.equal(denied.status, 401);
  assert.match(denied.headers.get("www-authenticate"), /Basic/);
  const allowed = await worker.fetch(new Request("https://example.com/dashboard", { headers: { authorization: `Basic ${btoa("radar:admin")}` } }), env);
  assert.equal(allowed.status, 200);
  const html = await allowed.text();
  assert.match(html, /Qué aporta/);
  assert.match(html, /Qué tomamos/);
  assert.match(html, /Qué descartamos/);
  assert.match(html, /Plan del día/);
  assert.doesNotMatch(html, /sender_hash/);
});
