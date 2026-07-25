from __future__ import annotations

from .models import (
    CommercialOpportunity,
    EditorialDecision,
    MeasurementPlan,
)


def build_measurement_plan(
    decision: EditorialDecision,
    opportunity: CommercialOpportunity | None,
) -> MeasurementPlan:
    if opportunity:
        primary_goal = "commercial_opportunity"
    elif decision.score >= 80:
        primary_goal = "authority"
    else:
        primary_goal = "audience_learning"
    return MeasurementPlan(
        primary_goal=primary_goal,
        pre_publish_checks=[
            "editorial_score",
            "source_count",
            "rights_status",
            "estimated_cost",
            "estimated_production_minutes",
        ],
        post_publish_metrics=[
            "retention_rate",
            "completion_rate",
            "share_rate",
            "save_rate",
            "useful_comment_rate",
            "qualified_follower_rate",
        ],
        commercial_metrics=[
            "qualified_leads",
            "proposal_requests",
            "proposals_sent",
            "revenue_attributed",
        ],
    )
