def render_editorial_asset(
    plan,
    *,
    render_clean_fn,
    render_gamer_fn,
    headline,
    image_path=None,
    video_bg_path=None,
    logo_path=None,
    seconds=8,
    music_path=None,
):
    style = plan.get("style_family", "editorial_clean")
    title_text = plan.get("title_text", headline)
    cta_text = plan.get("cta_text", "¿W o humo?")
    badge_text = plan.get("badge_text", "HOT")

    if style == "reel_gamer":
        return render_gamer_fn(
            headline=title_text,
            news_image_path=image_path,
            logo_path=logo_path,
            seconds=seconds,
            music_path=music_path,
            cta_text=cta_text,
            badge_text=badge_text,
        )

    if video_bg_path:
        return render_clean_fn(
            headline=title_text,
            bg_video_path=video_bg_path,
            logo_path=logo_path,
            seconds=seconds,
            music_path=music_path,
            cta_text=cta_text,
            badge_text=badge_text,
        )

    return render_gamer_fn(
        headline=title_text,
        news_image_path=image_path,
        logo_path=logo_path,
        seconds=seconds,
        music_path=music_path,
        cta_text=cta_text,
        badge_text=badge_text,
    )
