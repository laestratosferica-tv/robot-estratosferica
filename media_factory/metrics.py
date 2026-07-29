from __future__ import annotations

from .models import (
    CommercialOpportunity,
    EditorialDecision,
    MeasurementPlan,
)


def _rate(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return round(min(1.0, max(0.0, numerator / denominator)), 6)


def calculate_performance_metrics(snapshot: dict) -> dict:
    """Normalize cross-platform signals without rewarding views alone."""
    exposure = float(
        snapshot.get("reach")
        or snapshot.get("views")
        or snapshot.get("impressions")
        or 0
    )
    comments = float(snapshot.get("comments", 0)) + float(
        snapshot.get("replies", 0)
    )
    shares = float(snapshot.get("shares", 0)) + float(
        snapshot.get("reposts", 0)
    ) + float(snapshot.get("quotes", 0))
    saves = float(snapshot.get("saves", 0))
    follows = float(snapshot.get("follows", 0)) + float(
        snapshot.get("subscribers_gained", 0)
    )
    votes = float(snapshot.get("poll_votes", 0))
    qualified_answers = float(snapshot.get("qualified_answers", 0))
    commercial_signals = float(snapshot.get("commercial_signals", 0))

    conversation_rate = _rate(comments + votes, exposure)
    useful_answer_rate = _rate(qualified_answers, max(comments + votes, 1))
    share_rate = _rate(shares, exposure)
    save_rate = _rate(saves, exposure)
    qualified_follower_rate = _rate(follows, exposure)
    commercial_signal_rate = _rate(commercial_signals, exposure)
    completion_rate = float(
        snapshot.get("completion_rate")
        or snapshot.get("average_percentage_viewed")
        or 0
    )
    if completion_rate > 1:
        completion_rate /= 100

    learning_score = round(
        min(
            100.0,
            100
            * (
            conversation_rate * 0.25
            + useful_answer_rate * 0.20
            + share_rate * 0.20
            + save_rate * 0.15
            + qualified_follower_rate * 0.10
            + min(1.0, completion_rate) * 0.10
            ),
        ),
        3,
    )
    return {
        "exposure": int(exposure),
        "conversation_rate": conversation_rate,
        "useful_answer_rate": useful_answer_rate,
        "share_rate": share_rate,
        "save_rate": save_rate,
        "qualified_follower_rate": qualified_follower_rate,
        "commercial_signal_rate": commercial_signal_rate,
        "completion_rate": round(completion_rate, 6),
        "learning_score": learning_score,
        "views_are_not_the_primary_score": True,
    }


def summarize_audience_learning(records: list[dict]) -> dict:
    normalized = []
    for record in records:
        metrics = calculate_performance_metrics(record)
        normalized.append({**record, "normalized_metrics": metrics})
    ranked = sorted(
        normalized,
        key=lambda item: item["normalized_metrics"]["learning_score"],
        reverse=True,
    )
    return {
        "mode": "analysis_only",
        "records_analyzed": len(ranked),
        "best_learning_signal": ranked[0] if ranked else None,
        "ranked_experiments": ranked,
        "automatic_strategy_changes_enabled": False,
        "requires_human_review": True,
    }


def build_reel_learning_report(
    snapshot: dict,
    baseline: dict | None = None,
) -> dict:
    """Diagnose one Reel without claiming success from incomplete public data."""
    metrics = calculate_performance_metrics(snapshot)
    baseline_metrics = calculate_performance_metrics(baseline or {})
    available = {
        key
        for key, value in snapshot.items()
        if value is not None
    }
    required = {
        "views",
        "completion_rate",
        "shares",
        "saves",
        "comments",
    }
    missing = sorted(required - available)
    comparable = bool(baseline) and not missing

    if "completion_rate" not in available:
        diagnosis = "insufficient_retention_data"
        next_test = {
            "variable": "none",
            "action": "collect_owner_insights",
            "reason": "La reproducción pública no revela dónde abandona la audiencia.",
        }
    elif metrics["completion_rate"] < 0.45:
        diagnosis = "weak_completion"
        next_test = {
            "variable": "hook",
            "action": "test_clearer_first_second",
            "reason": "La finalización baja exige aclarar antes la promesa.",
        }
    elif metrics["share_rate"] < 0.01 and metrics["save_rate"] < 0.01:
        diagnosis = "watched_but_not_shared"
        next_test = {
            "variable": "payoff",
            "action": "test_more_useful_or_surprising_payoff",
            "reason": "La pieza se ve, pero todavía no genera suficiente valor social.",
        }
    else:
        diagnosis = "promising_quality_signals"
        next_test = {
            "variable": "hook",
            "action": "preserve_structure_test_new_hook",
            "reason": "Se conserva la estructura y se prueba una sola entrada nueva.",
        }

    exposure_delta = None
    if comparable and baseline_metrics["exposure"] > 0:
        exposure_delta = round(
            (
                metrics["exposure"] - baseline_metrics["exposure"]
            )
            / baseline_metrics["exposure"],
            4,
        )

    return {
        "mode": "analysis_only",
        "diagnosis": diagnosis,
        "metrics": metrics,
        "baseline_metrics": baseline_metrics if baseline else None,
        "exposure_delta": exposure_delta,
        "missing_metrics": missing,
        "data_quality": "complete" if not missing else "partial",
        "next_test": next_test,
        "one_variable_only": True,
        "automatic_strategy_changes_enabled": False,
        "requires_human_review": True,
    }


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
            "conversation_rate",
            "poll_participation_rate",
            "qualified_answers",
            "qualified_follower_rate",
        ],
        commercial_metrics=[
            "qualified_leads",
            "commercial_signal_rate",
            "sponsor_fit_signals",
            "proposal_requests",
            "proposals_sent",
            "revenue_attributed",
        ],
    )
