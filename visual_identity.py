from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict


BRAND_DNA = {
    "name": "La Estratosférica",
    "short_mark": "LETV",
    "symbol": "orbital_arc",
    "core_gradient": ["#19D9FF", "#6D5CFF", "#FF3E9D"],
    "base": "#070812",
    "light": "#F7F6F2",
    "signature_weights": {
        "full": 60,
        "short": 20,
        "none": 20,
    },
    "mix": {
        "brand_dna": 70,
        "category_language": 20,
        "trend_signal": 10,
    },
}


CATEGORY_DIRECTIONS = {
    "gaming": {
        "accent": "#E9F056",
        "secondary": "#3A66FF",
        "layout": "kinetic",
        "temperature": "electric_cool",
        "contrast": 1.16,
        "saturation": 1.05,
        "blur_mix": 0.10,
        "headline_scale": 1.04,
        "texture": "speed_marks",
        "background_mode": "full_bleed",
        "focal_anchor": "center",
        "headline_anchor": "bottom",
        "photo_scale": 1.10,
        "tint_strength": 0.10,
    },
    "technology": {
        "accent": "#D7EFFF",
        "secondary": "#19D9FF",
        "layout": "precision",
        "temperature": "chrome_cool",
        "contrast": 1.10,
        "saturation": 0.82,
        "blur_mix": 0.14,
        "headline_scale": 0.96,
        "texture": "data_grid",
        "background_mode": "split_focus",
        "focal_anchor": "right",
        "headline_anchor": "bottom_left",
        "photo_scale": 1.00,
        "tint_strength": 0.18,
    },
    "advertising": {
        "accent": "#FF3E9D",
        "secondary": "#FFD35A",
        "layout": "editorial_play",
        "temperature": "creative_warm",
        "contrast": 1.12,
        "saturation": 1.06,
        "blur_mix": 0.12,
        "headline_scale": 1.00,
        "texture": "crop_marks",
        "background_mode": "duotone_collage",
        "focal_anchor": "left",
        "headline_anchor": "bottom",
        "photo_scale": 1.08,
        "tint_strength": 0.26,
    },
    "fashion": {
        "accent": "#EEE9E1",
        "secondary": "#8C62FF",
        "layout": "editorial",
        "temperature": "neutral_plum",
        "contrast": 1.06,
        "saturation": 0.88,
        "blur_mix": 0.08,
        "headline_scale": 0.92,
        "texture": "fine_grain",
        "background_mode": "editorial_inset",
        "focal_anchor": "top",
        "headline_anchor": "bottom_left",
        "photo_scale": 0.84,
        "tint_strength": 0.20,
    },
    "gastronomy": {
        "accent": "#FF6A3D",
        "secondary": "#B8C0A8",
        "layout": "sensory",
        "temperature": "warm",
        "contrast": 1.04,
        "saturation": 1.12,
        "blur_mix": 0.06,
        "headline_scale": 0.96,
        "texture": "soft_glow",
        "background_mode": "warm_focus",
        "focal_anchor": "center",
        "headline_anchor": "bottom",
        "photo_scale": 1.04,
        "tint_strength": 0.12,
    },
    "lifestyle": {
        "accent": "#B8C0A8",
        "secondary": "#D7EFFF",
        "layout": "open",
        "temperature": "natural_cool",
        "contrast": 1.02,
        "saturation": 0.94,
        "blur_mix": 0.10,
        "headline_scale": 0.92,
        "texture": "soft_frame",
        "background_mode": "soft_canvas",
        "focal_anchor": "center",
        "headline_anchor": "bottom_left",
        "photo_scale": 0.90,
        "tint_strength": 0.16,
    },
    "luxury": {
        "accent": "#E9E1D2",
        "secondary": "#9D8CFF",
        "layout": "spacious",
        "temperature": "plum_noir",
        "contrast": 1.08,
        "saturation": 0.72,
        "blur_mix": 0.05,
        "headline_scale": 0.86,
        "texture": "hairline",
        "background_mode": "gallery_window",
        "focal_anchor": "center",
        "headline_anchor": "bottom_left",
        "photo_scale": 0.76,
        "tint_strength": 0.24,
    },
    "monetization": {
        "accent": "#E9F056",
        "secondary": "#19D9FF",
        "layout": "signal",
        "temperature": "optimistic_cool",
        "contrast": 1.12,
        "saturation": 0.98,
        "blur_mix": 0.12,
        "headline_scale": 0.98,
        "texture": "signal_bars",
        "background_mode": "signal_split",
        "focal_anchor": "right",
        "headline_anchor": "bottom_left",
        "photo_scale": 0.96,
        "tint_strength": 0.18,
    },
}


TREND_PROFILES = {
    "evergreen": {
        "name": "Evergreen digital editorial",
        "accent": "#D7EFFF",
        "texture": "none",
        "valid_from": None,
        "valid_until": None,
    },
    "sport_luxe_2026_q3": {
        "name": "Sport-Luxe Digital",
        "accent": "#E9F056",
        "texture": "motorsport_hairlines",
        "valid_from": "2026-07-01",
        "valid_until": "2026-09-30",
    },
}


def get_visual_direction(
    pillar: str,
    trend_profile: str = "evergreen",
) -> Dict[str, Any]:
    safe_pillar = pillar if pillar in CATEGORY_DIRECTIONS else "gaming"
    safe_trend = trend_profile if trend_profile in TREND_PROFILES else "evergreen"
    direction = deepcopy(CATEGORY_DIRECTIONS[safe_pillar])
    direction.update(
        {
            "pillar": safe_pillar,
            "brand": deepcopy(BRAND_DNA),
            "trend": deepcopy(TREND_PROFILES[safe_trend]),
            "trend_profile": safe_trend,
        }
    )
    return direction


def validate_visual_system() -> None:
    if sum(BRAND_DNA["mix"].values()) != 100:
        raise ValueError("Visual identity mix must total 100")
    if BRAND_DNA["mix"]["brand_dna"] < 60:
        raise ValueError("Brand DNA must remain the dominant layer")
    if BRAND_DNA["mix"]["trend_signal"] > 10:
        raise ValueError("Trend signal cannot overpower the identity")
    if len(BRAND_DNA["core_gradient"]) != 3:
        raise ValueError("The orbital gradient requires three stops")
    if sum(BRAND_DNA["signature_weights"].values()) != 100:
        raise ValueError("Brand signature weights must total 100")
    if BRAND_DNA["signature_weights"]["full"] <= BRAND_DNA["signature_weights"]["short"]:
        raise ValueError("The full brand name must appear more often than LETV")
    required = {
        "accent",
        "secondary",
        "layout",
        "temperature",
        "contrast",
        "saturation",
        "blur_mix",
        "headline_scale",
        "texture",
        "background_mode",
        "focal_anchor",
        "headline_anchor",
        "photo_scale",
        "tint_strength",
    }
    for pillar, direction in CATEGORY_DIRECTIONS.items():
        missing = required.difference(direction)
        if missing:
            raise ValueError(f"{pillar} is missing: {sorted(missing)}")


validate_visual_system()
