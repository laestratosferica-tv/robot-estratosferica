import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const publicRoot = new URL("../public/", import.meta.url);

test("publishes crawl controls and a canonical sitemap", async () => {
  const [robots, sitemap] = await Promise.all([
    readFile(new URL("robots.txt", publicRoot), "utf8"),
    readFile(new URL("sitemap.xml", publicRoot), "utf8"),
  ]);
  assert.match(robots, /Sitemap: https:\/\/promodetector\.co\/sitemap\.xml/);
  assert.match(robots, /Disallow: \/batch-review\.html/);
  assert.match(sitemap, /<loc>https:\/\/promodetector\.co\/<\/loc>/);
  assert.doesNotMatch(sitemap, /batch-review/);
});

test("catalog status is fail-closed and scores use ten-point scale", async () => {
  const status = JSON.parse(await readFile(new URL("catalog-status.json", publicRoot), "utf8"));
  assert.equal(status.publication_mode, "fail_closed");
  assert.ok(status.totals.requires_revalidation > 0);
  for (const item of status.items) {
    if (item.score !== undefined) assert.ok(item.score >= 0 && item.score <= 10);
  }
});

test("product pages expose canonical editorial review data", async () => {
  const html = await readFile(new URL("mini-mic-pro.html", publicRoot), "utf8");
  assert.match(html, /rel="canonical" href="https:\/\/promodetector\.co\/mini-mic-pro\.html"/);
  assert.match(html, /"@type":"Product"/);
  assert.match(html, /"@type":"Review"/);
  assert.match(html, /"bestRating":10/);
  assert.doesNotMatch(html, /"price"|"availability"/);
});
