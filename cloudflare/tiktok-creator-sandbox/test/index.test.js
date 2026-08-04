import assert from "node:assert/strict";
import test from "node:test";
import worker from "../src/index.js";

test("health proves every external write is disabled by default", async () => {
  const response = await worker.fetch(new Request("https://example.com/health"), {});
  const body = await response.json();
  assert.equal(body.ok, true);
  assert.equal(body.draft_transfer_enabled, false);
  assert.equal(body.transfer_allowed, false);
  assert.equal(body.direct_post_enabled, false);
});

test("public page explains an external creator product", async () => {
  const response = await worker.fetch(new Request("https://example.com/"), {});
  const body = await response.text();
  assert.match(body, /Portal para creadores/u);
  assert.match(body, /Tu contenido/u);
  assert.match(body, /La Estratosférica nunca publica por ti/u);
});

test("oauth callback follows the public request origin", async () => {
  const response = await worker.fetch(new Request("https://public-example.workers.dev/oauth/tiktok/start"), {
    TIKTOK_CLIENT_KEY: "key",
    TIKTOK_CLIENT_SECRET: "secret",
    SESSION_SECRET: "session-secret",
  });
  const location = new URL(response.headers.get("location"));
  assert.equal(location.searchParams.get("redirect_uri"), "https://public-example.workers.dev/oauth/tiktok/callback");
  assert.equal(location.searchParams.get("scope"), "user.info.basic,video.upload");
});
