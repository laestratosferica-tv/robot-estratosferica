import { readFile, writeFile, access } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

const root = path.resolve(import.meta.dirname, "..");
const publicDir = path.join(root, "public");
const catalogPath = path.join(root, "data", "catalog.json");
const now = process.env.PROMODETECTOR_NOW
  ? new Date(process.env.PROMODETECTOR_NOW)
  : new Date();

const catalog = JSON.parse(await readFile(catalogPath, "utf8"));
const site = catalog.site.replace(/\/$/, "");
const freshnessMs = catalog.freshness_hours * 60 * 60 * 1000;

function escapeXml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&apos;");
}

function assertItem(item) {
  for (const field of ["slug", "kind", "title", "description", "category", "page", "source_url", "source_type", "market", "last_verified_at", "status"]) {
    if (!item[field]) throw new Error(`${item.slug ?? "item"}: missing ${field}`);
  }
  if (!item.source_url.startsWith("https://")) throw new Error(`${item.slug}: source_url must use HTTPS`);
  if (item.score !== undefined && (item.score < 0 || item.score > 10)) throw new Error(`${item.slug}: score must be 0-10`);
}

for (const item of catalog.items) {
  assertItem(item);
  await access(path.join(publicDir, item.page));
  if (item.image) await access(path.join(publicDir, item.image));
}

const assessed = catalog.items.map((item) => {
  const ageMs = now.getTime() - new Date(item.last_verified_at).getTime();
  const evergreen = item.status === "evergreen";
  const fresh = evergreen || (ageMs >= 0 && ageMs <= freshnessMs);
  return {
    ...item,
    fresh,
    active: fresh && ["verified", "evergreen"].includes(item.status),
    requires_revalidation: !evergreen && !fresh,
  };
});

const indexable = assessed.filter((item) => item.kind === "editorial_guide" || item.kind === "product_review");
const status = {
  schema: "promodetector_catalog_status_v1",
  generated_at: now.toISOString(),
  publication_mode: "fail_closed",
  totals: {
    catalog: assessed.length,
    active: assessed.filter((item) => item.active).length,
    requires_revalidation: assessed.filter((item) => item.requires_revalidation).length,
    indexable: indexable.length,
  },
  items: assessed.map(({ source_url, ...item }) => item),
};

const urls = [
  { loc: `${site}/`, lastmod: now.toISOString().slice(0, 10), priority: "1.0" },
  ...indexable.map((item) => ({
    loc: `${site}/${item.page}`,
    lastmod: item.last_verified_at.slice(0, 10),
    priority: item.kind === "product_review" ? "0.8" : "0.7",
  })),
];

const sitemap = `<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n${urls
  .map((url) => `  <url><loc>${escapeXml(url.loc)}</loc><lastmod>${url.lastmod}</lastmod><changefreq>daily</changefreq><priority>${url.priority}</priority></url>`)
  .join("\n")}\n</urlset>\n`;

const feedItems = indexable
  .map((item) => `    <item><title>${escapeXml(item.title)}</title><link>${site}/${escapeXml(item.page)}</link><guid>${site}/${escapeXml(item.page)}</guid><description>${escapeXml(item.description)}</description><pubDate>${new Date(item.last_verified_at).toUTCString()}</pubDate></item>`)
  .join("\n");
const feed = `<?xml version="1.0" encoding="UTF-8"?>\n<rss version="2.0"><channel><title>PromoDetector</title><link>${site}</link><description>Señales, comparaciones y guías verificadas de tecnología y gaming.</description><language>es</language>\n${feedItems}\n</channel></rss>\n`;

function seoBlock(item) {
  const url = `${site}/${item.page}`;
  const image = item.image ? `${site}/${item.image}` : `${site}/og.png`;
  const structured = item.kind === "product_review"
    ? {
        "@context": "https://schema.org",
        "@type": "Product",
        name: item.title,
        image,
        description: item.description,
        review: {
          "@type": "Review",
          name: `Valoración PromoDetector: ${item.title}`,
          author: { "@type": "Organization", name: "La Estratosférica" },
          reviewRating: { "@type": "Rating", ratingValue: item.score, bestRating: 10, worstRating: 0 },
        },
      }
    : {
        "@context": "https://schema.org",
        "@type": "Article",
        headline: item.title,
        description: item.description,
        mainEntityOfPage: url,
        dateModified: item.last_verified_at,
        publisher: { "@type": "Organization", name: "La Estratosférica" },
      };
  return `<!-- PROMODETECTOR_SEO_START -->\n<meta name="robots" content="index,follow,max-image-preview:large">\n<link rel="canonical" href="${url}">\n<meta property="og:type" content="article">\n<meta property="og:title" content="${escapeXml(item.title)} — PromoDetector">\n<meta property="og:description" content="${escapeXml(item.description)}">\n<meta property="og:url" content="${url}">\n<meta property="og:image" content="${image}">\n<script type="application/ld+json">${JSON.stringify(structured).replaceAll("<", "\\u003c")}</script>\n<!-- PROMODETECTOR_SEO_END -->`;
}

for (const item of indexable) {
  const pagePath = path.join(publicDir, item.page);
  let html = await readFile(pagePath, "utf8");
  html = html.replace(/<!-- PROMODETECTOR_SEO_START -->[\s\S]*?<!-- PROMODETECTOR_SEO_END -->/g, "");
  if (!html.includes("</head>")) throw new Error(`${item.page}: missing </head>`);
  html = html.replace("</head>", `${seoBlock(item)}</head>`);
  await writeFile(pagePath, html, "utf8");
}

await Promise.all([
  writeFile(path.join(publicDir, "catalog-status.json"), `${JSON.stringify(status, null, 2)}\n`, "utf8"),
  writeFile(path.join(publicDir, "sitemap.xml"), sitemap, "utf8"),
  writeFile(path.join(publicDir, "feed.xml"), feed, "utf8"),
  writeFile(path.join(publicDir, "robots.txt"), `User-agent: *\nAllow: /\nDisallow: /batch-review.html\nSitemap: ${site}/sitemap.xml\n`, "utf8"),
]);

console.log(JSON.stringify(status.totals));
if (process.argv.includes("--require-fresh") && status.totals.active === 0) {
  console.error("No fresh verified commercial signals. Publication remains blocked.");
  process.exitCode = 2;
}
