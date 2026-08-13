from __future__ import annotations

import argparse
import json
from pathlib import Path

from media_factory.production_director import build_production_request, load_cast


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepara una orden HeyGen sin renderizar")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/private/heygen-job.json")
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    request = build_production_request(payload, load_cast(ROOT / "config/virtual_cast_v2.json"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(request.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"state": request.state, "character": request.character_name, "blockers": request.blockers}, ensure_ascii=False))
    return 0 if request.state == "ready_for_heygen" else 2


if __name__ == "__main__":
    raise SystemExit(main())
