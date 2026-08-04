const API_VERSION = "2022-11-28";
const RECENT_RUN_WINDOW_MS = 4 * 60 * 1000;

function githubHeaders(token) {
  return {
    Accept: "application/vnd.github+json",
    Authorization: `Bearer ${token}`,
    "X-GitHub-Api-Version": API_VERSION,
    "User-Agent": "estratosferica-cloudflare-scheduler",
  };
}

export function shouldDispatch(runs, now = Date.now()) {
  return !runs.some((run) => {
    if (run.status === "queued" || run.status === "in_progress") return true;
    const createdAt = Date.parse(run.created_at || "");
    return Number.isFinite(createdAt) && now - createdAt < RECENT_RUN_WINDOW_MS;
  });
}

async function listRecentRuns(env) {
  const url = `https://api.github.com/repos/${env.GITHUB_OWNER}/${env.GITHUB_REPO}/actions/workflows/${env.GITHUB_WORKFLOW}/runs?per_page=5`;
  const response = await fetch(url, { headers: githubHeaders(env.GITHUB_TOKEN) });
  if (!response.ok) {
    throw new Error(`GitHub runs lookup failed: ${response.status}`);
  }
  const payload = await response.json();
  return Array.isArray(payload.workflow_runs) ? payload.workflow_runs : [];
}

async function dispatchWorkflow(env) {
  const url = `https://api.github.com/repos/${env.GITHUB_OWNER}/${env.GITHUB_REPO}/actions/workflows/${env.GITHUB_WORKFLOW}/dispatches`;
  const response = await fetch(url, {
    method: "POST",
    headers: {
      ...githubHeaders(env.GITHUB_TOKEN),
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ ref: env.GITHUB_REF, inputs: { live: "true" } }),
  });
  if (response.status !== 204) {
    throw new Error(`GitHub workflow dispatch failed: ${response.status}`);
  }
}

export async function runScheduler(env, now = Date.now()) {
  if (!env.GITHUB_TOKEN) throw new Error("GITHUB_TOKEN is not configured");
  const runs = await listRecentRuns(env);
  if (!shouldDispatch(runs, now)) return { status: "skipped", reason: "recent-or-active-run" };
  await dispatchWorkflow(env);
  return { status: "dispatched" };
}

export default {
  async scheduled(_controller, env, ctx) {
    ctx.waitUntil(runScheduler(env));
  },
  async fetch(request) {
    const url = new URL(request.url);
    if (url.pathname !== "/health") return new Response("Not found", { status: 404 });
    return Response.json({ ok: true, service: "estratosferica-publisher-scheduler" });
  },
};

