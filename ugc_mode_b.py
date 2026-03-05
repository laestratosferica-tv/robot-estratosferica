def ig_api_post(path, data):
    r = requests.post(f"{GRAPH_BASE}/{path}", data=data, timeout=HTTP_TIMEOUT)
    try:
        j = r.json()
    except Exception:
        j = {"raw": (r.text or "")[:2000]}
    if r.status_code >= 400:
        raise RuntimeError(f"IG POST {path} failed {r.status_code}: {j}")
    return j

def ig_api_get(path, params):
    r = requests.get(f"{GRAPH_BASE}/{path}", params=params, timeout=HTTP_TIMEOUT)
    try:
        j = r.json()
    except Exception:
        j = {"raw": (r.text or "")[:2000]}
    if r.status_code >= 400:
        raise RuntimeError(f"IG GET {path} failed {r.status_code}: {j}")
    return j

def ig_wait_container(creation_id, timeout_sec=900):
    start = time.time()
    while time.time() - start < timeout_sec:
        j = ig_api_get(
            f"{creation_id}",
            {"fields": "status_code,status", "access_token": IG_ACCESS_TOKEN},
        )
        status = (j.get("status_code") or j.get("status") or "").upper()
        print("IG container status:", creation_id, status)
        if status in ("FINISHED", "PUBLISHED"):
            return
        if status in ("ERROR", "FAILED"):
            raise RuntimeError(f"IG container FAILED: {j}")
        time.sleep(3)
    raise TimeoutError(f"IG container timeout after {timeout_sec}s: {creation_id}")

def ig_publish(video_url, caption):
    if UGC_DRY_RUN:
        print("[DRY_RUN] IG publish skipped:", video_url)
        return {"ok": True, "dry_run": True, "video_url": video_url}

    if not IG_USER_ID or not IG_ACCESS_TOKEN:
        raise RuntimeError("Faltan IG_USER_ID o IG_ACCESS_TOKEN")

    print("IG create container:", video_url)
    create = ig_api_post(
        f"{IG_USER_ID}/media",
        {
            "media_type": "REELS",
            "video_url": video_url,
            "caption": caption,
            "share_to_feed": "true",
            "access_token": IG_ACCESS_TOKEN,
        },
    )
    print("IG create response:", create)

    creation_id = create.get("id")
    if not creation_id:
        raise RuntimeError(f"IG create did not return id: {create}")

    ig_wait_container(creation_id, timeout_sec=900)

    print("IG publish container:", creation_id)
    pub = ig_api_post(
        f"{IG_USER_ID}/media_publish",
        {"creation_id": creation_id, "access_token": IG_ACCESS_TOKEN},
    )
    print("IG publish response:", pub)
    return {"ok": True, "creation_id": creation_id, "publish": pub, "video_url": video_url}
