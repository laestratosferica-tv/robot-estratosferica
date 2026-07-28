from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Candidate:
    title: str
    source_url: str
    territory: str
    candidate_id: str = ""
    summary: str = ""
    source_id: str = ""
    published_at: str = ""
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
            candidate_id=str(data.get("candidate_id", "")).strip(),
            summary=str(data.get("summary", "")).strip(),
            source_id=str(data.get("source_id", "")).strip(),
            published_at=str(data.get("published_at", "")).strip(),
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


@dataclass(frozen=True)
class CommercialOpportunity:
    kind: str
    score: int
    status: str
    rationale: str
    next_step: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MeasurementPlan:
    primary_goal: str
    pre_publish_checks: list[str]
    post_publish_metrics: list[str]
    commercial_metrics: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ContentPackage:
    format_id: str
    state: str
    headline: str
    angle: str
    factual_summary: str
    short_video_script: str
    platform_copy: dict[str, str]
    visual_brief: list[str]
    sources: list[str]
    talent: dict[str, str] = field(default_factory=dict)
    audience_experiment: dict[str, Any] = field(default_factory=dict)
    content_punch: dict[str, Any] = field(default_factory=dict)
    requires_human_review: bool = True
    external_actions_enabled: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StoryboardScene:
    scene_id: str
    start_second: int
    end_second: int
    purpose: str
    voiceover: str
    on_screen_text: str
    visual_direction: str
    audio_direction: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Storyboard:
    state: str
    master_format: str
    duration_seconds: int
    frames_per_second: int
    captions_required: bool
    visual_style: list[str]
    scenes: list[StoryboardScene]
    source_card: dict[str, str]
    requires_human_review: bool = True
    production_enabled: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "scenes": [scene.to_dict() for scene in self.scenes],
        }


@dataclass(frozen=True)
class PipelineItem:
    candidate: Candidate
    decision: EditorialDecision
    measurement_plan: MeasurementPlan
    commercial_opportunity: CommercialOpportunity | None = None
    content_package: ContentPackage | None = None
    storyboard: Storyboard | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "story": {
                "candidate_id": self.candidate.candidate_id,
                "title": self.candidate.title,
                "source_url": self.candidate.source_url,
                "source_id": self.candidate.source_id,
                "published_at": self.candidate.published_at,
                "territory": self.candidate.territory,
                "region": self.candidate.region,
            },
            "decision": self.decision.to_dict(),
            "measurement_plan": self.measurement_plan.to_dict(),
            "commercial_opportunity": (
                self.commercial_opportunity.to_dict()
                if self.commercial_opportunity
                else None
            ),
            "content_package": (
                self.content_package.to_dict()
                if self.content_package
                else None
            ),
            "storyboard": (
                self.storyboard.to_dict() if self.storyboard else None
            ),
        }
