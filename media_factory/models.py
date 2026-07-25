from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Candidate:
    title: str
    source_url: str
    territory: str
    region: str = "latam"
    is_duplicate: bool = False
    is_verified: bool = True
    has_media_rights: bool = True
    claims_supported: bool = True
    signals: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Candidate":
        return cls(
            title=str(data.get("title", "")).strip(),
            source_url=str(data.get("source_url", "")).strip(),
            territory=str(data.get("territory", "")).strip(),
            region=str(data.get("region", "latam")).strip(),
            is_duplicate=bool(data.get("is_duplicate", False)),
            is_verified=bool(data.get("is_verified", True)),
            has_media_rights=bool(data.get("has_media_rights", True)),
            claims_supported=bool(data.get("claims_supported", True)),
            signals=dict(data.get("signals", {})),
        )


@dataclass(frozen=True)
class EditorialDecision:
    title: str
    score: int
    state: str
    accepted: bool
    rejection_reasons: list[str]
    score_breakdown: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
