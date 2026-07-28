from __future__ import annotations

import argparse
import html
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import parse_qs

ALLOWED_DECISIONS = {"approved_editorially", "rejected"}


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_queue(queue: Mapping[str, Any]) -> None:
    errors: list[str] = []
    if queue.get("schema_version") != "review_queue_v1":
        errors.append("schema_version inválido")
    if queue.get("mode") != "dry_run":
        errors.append("la cola debe permanecer en dry_run")
    if queue.get("publishing_enabled") is not False:
        errors.append("publishing_enabled debe ser false")
    if queue.get("external_actions_enabled") is not False:
        errors.append("external_actions_enabled debe ser false")
    if queue.get("human_approval_required") is not True:
        errors.append("la aprobación humana debe ser obligatoria")
    if len(queue.get("items", [])) > 1:
        errors.append("el panel V1 admite máximo una pieza por corrida")
    for item in queue.get("items", []):
        review = item.get("review", {})
        if review.get("status") != "pending_human_approval":
            errors.append("la pieza debe estar pendiente de aprobación")
        if review.get("publish_allowed") is not False:
            errors.append("la pieza no puede habilitar publicación")
        if review.get("requires_human_approval") is not True:
            errors.append("la pieza debe exigir aprobación humana")
    if errors:
        raise ValueError("; ".join(errors))


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def record_decision(
    queue: Mapping[str, Any],
    decisions_path: str | Path,
    *,
    review_id: str,
    decision: str,
    reason: str = "",
    reviewer: str = "José Luis",
    now: datetime | None = None,
) -> dict[str, Any]:
    validate_queue(queue)
    if decision not in ALLOWED_DECISIONS:
        raise ValueError("decisión no permitida")
    if decision == "rejected" and not reason.strip():
        raise ValueError("el rechazo requiere un motivo")

    matching = [
        item for item in queue.get("items", [])
        if item.get("review", {}).get("review_id") == review_id
    ]
    if len(matching) != 1:
        raise ValueError("review_id desconocido o duplicado")

    target = Path(decisions_path)
    ledger = (
        load_json(target)
        if target.exists()
        else {
            "schema_version": "editorial_decisions_v1",
            "publishing_enabled": False,
            "external_actions_enabled": False,
            "decisions": [],
        }
    )
    if ledger.get("schema_version") != "editorial_decisions_v1":
        raise ValueError("bitácora de decisiones incompatible")
    if ledger.get("publishing_enabled") is not False:
        raise ValueError("la bitácora no puede habilitar publicación")
    if ledger.get("external_actions_enabled") is not False:
        raise ValueError("la bitácora no puede habilitar acciones externas")
    if any(
        entry.get("review_id") == review_id
        for entry in ledger.get("decisions", [])
    ):
        raise ValueError("esta pieza ya tiene una decisión registrada")

    review = matching[0]["review"]
    timestamp = (now or datetime.now(timezone.utc)).isoformat()
    entry = {
        "decision_id": f"{review_id}:{timestamp}",
        "review_id": review_id,
        "candidate_id": review["candidate_id"],
        "content_fingerprint": review["content_fingerprint"],
        "decision": decision,
        "reason": reason.strip(),
        "reviewer": reviewer.strip() or "revisor_humano",
        "decided_at": timestamp,
        "editorial_approval_only": decision == "approved_editorially",
        "publish_allowed": False,
        "publishing_enabled": False,
        "external_actions_enabled": False,
    }
    ledger.setdefault("decisions", []).append(entry)
    _atomic_write(target, ledger)
    return entry


def _e(value: Any) -> str:
    return html.escape(str(value or ""), quote=True)


def render_panel(
    queue: Mapping[str, Any],
    decisions: Mapping[str, Any] | None = None,
) -> str:
    validate_queue(queue)
    recorded = {
        entry.get("review_id"): entry
        for entry in (decisions or {}).get("decisions", [])
    }
    cards: list[str] = []
    for item in queue.get("items", []):
        review = item["review"]
        story = item.get("story", {})
        selection = review.get("opportunity_selection", {})
        test = review.get("editorial_test", {})
        strategy = review.get("strategy", {})
        existing = recorded.get(review["review_id"])
        copies = "".join(
            f"<section><h3>{_e(platform)}</h3><p>{_e(copy)}</p></section>"
            for platform, copy in review.get(
                "final_text_by_platform", {}
            ).items()
        ) or "<p>No hay textos preparados.</p>"
        options = "".join(
            f"<li>{_e(option)}</li>"
            for option in test.get("answer_options", [])
        )
        if existing:
            action = (
                "<div class='decision'>Decisión registrada: "
                f"<strong>{_e(existing['decision'])}</strong>"
                f"<p>{_e(existing.get('reason'))}</p></div>"
            )
        else:
            action = f"""
            <form method="post" action="/decide">
              <input type="hidden" name="review_id"
                     value="{_e(review['review_id'])}">
              <label>Motivo del rechazo o nota editorial</label>
              <textarea name="reason" rows="3"></textarea>
              <div class="actions">
                <button class="approve" name="decision"
                        value="approved_editorially">
                  Aprobar editorialmente
                </button>
                <button class="reject" name="decision" value="rejected">
                  Rechazar
                </button>
              </div>
            </form>
            """
        cards.append(
            f"""
            <article class="card">
              <span class="badge">Pieza #{_e(selection.get('rank', 1))}</span>
              <h2>{_e(story.get('title'))}</h2>
              <p class="summary">{_e(item.get('content_package', {}).get('factual_summary'))}</p>
              <div class="grid">
                <section><h3>Objetivo</h3><p>{_e(test.get('objective'))}</p></section>
                <section><h3>Interacción</h3><p>{_e(test.get('expected_interaction'))}</p></section>
                <section><h3>Métrica principal</h3><p>{_e(test.get('primary_metric'))}</p></section>
                <section><h3>Producto</h3><p>{_e(strategy.get('content_product_id'))}</p></section>
              </div>
              <section>
                <h3>Pregunta para la comunidad</h3>
                <p>{_e(test.get('interaction_prompt'))}</p>
                <ul>{options}</ul>
              </section>
              <section>
                <h3>Razón de selección</h3>
                <p>Puntaje: {_e(selection.get('score'))}/100</p>
                <p>{_e(' · '.join(selection.get('rationale', [])))}</p>
              </section>
              <details><summary>Textos por plataforma</summary>{copies}</details>
              <p><a href="{_e(review['source']['url'])}" target="_blank"
                    rel="noopener noreferrer">Abrir fuente original</a></p>
              {action}
            </article>
            """
        )
    body = "".join(cards) or (
        "<article class='card'><h2>No hay piezas pendientes</h2>"
        "<p>El selector no envió una oportunidad a revisión.</p></article>"
    )
    return f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Revisión editorial · La Estratosférica</title>
  <style>
    :root {{ color-scheme: dark; --ink:#f6f7fb; --muted:#a8adc2;
      --panel:#141728; --line:#2b304b; --cyan:#4ce5ff; --pink:#ff4fa3; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font:16px/1.5 system-ui,sans-serif; color:var(--ink);
      background:radial-gradient(circle at top,#252a51,#090b13 55%);
      min-height:100vh; }}
    main {{ width:min(940px,92vw); margin:0 auto; padding:48px 0 80px; }}
    h1 {{ margin:.2rem 0; font-size:clamp(2rem,6vw,4rem); line-height:1; }}
    h2 {{ font-size:clamp(1.5rem,3vw,2.3rem); }}
    h3 {{ color:var(--cyan); font-size:.85rem; text-transform:uppercase;
      letter-spacing:.08em; margin-bottom:.3rem; }}
    p {{ color:var(--muted); }} .safe {{ color:#7effb2; }}
    .card {{ background:rgba(20,23,40,.94); border:1px solid var(--line);
      border-radius:22px; padding:clamp(20px,4vw,40px);
      box-shadow:0 24px 80px #0008; }}
    .badge {{ display:inline-block; color:#081017; background:var(--cyan);
      border-radius:999px; padding:4px 12px; font-weight:800; }}
    .summary {{ font-size:1.1rem; }}
    .grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; }}
    section {{ padding:12px 0; }}
    details {{ border-top:1px solid var(--line); border-bottom:1px solid var(--line);
      padding:15px 0; }}
    summary {{ cursor:pointer; font-weight:800; }}
    textarea {{ width:100%; margin:8px 0 14px; border-radius:12px;
      border:1px solid var(--line); background:#0d0f1b; color:var(--ink); padding:12px; }}
    label {{ display:block; color:var(--muted); margin-top:18px; }}
    .actions {{ display:flex; gap:12px; flex-wrap:wrap; }}
    button {{ border:0; border-radius:12px; padding:13px 18px;
      font-weight:850; cursor:pointer; }}
    .approve {{ background:var(--cyan); color:#071019; }}
    .reject {{ background:var(--pink); color:white; }}
    .decision {{ border:1px solid #7effb2; border-radius:12px; padding:16px;
      color:#7effb2; margin-top:18px; }}
    a {{ color:var(--cyan); }}
    @media(max-width:640px) {{ .grid {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body><main>
  <header>
    <p>LA ESTRATOSFÉRICA · CONTROL HUMANO</p>
    <h1>Revisión editorial</h1>
    <p class="safe">Modo seguro: aprobar aquí nunca publica.</p>
  </header>
  {body}
</main></body></html>"""


def create_app(
    queue_path: str | Path,
    decisions_path: str | Path,
) -> Any:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import HTMLResponse, RedirectResponse

    queue_file = Path(queue_path)
    decisions_file = Path(decisions_path)
    app = FastAPI(title="La Estratosférica · Revisión editorial")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        queue = load_json(queue_file)
        decisions = (
            load_json(decisions_file)
            if decisions_file.exists()
            else None
        )
        return render_panel(queue, decisions)

    @app.post("/decide")
    async def decide(request: Request) -> Any:
        form = parse_qs((await request.body()).decode("utf-8"))
        review_id = form.get("review_id", [""])[0]
        decision = form.get("decision", [""])[0]
        reason = form.get("reason", [""])[0]
        try:
            record_decision(
                load_json(queue_file),
                decisions_file,
                review_id=review_id,
                decision=decision,
                reason=reason,
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return RedirectResponse("/", status_code=303)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Panel local de aprobación editorial segura."
    )
    parser.add_argument("--queue", default="artifacts/editorial_queue.json")
    parser.add_argument(
        "--decisions", default="artifacts/editorial_decisions.json"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(
        create_app(args.queue, args.decisions),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
