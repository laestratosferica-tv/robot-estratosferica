{
  "accounts": [
    {
      "account_id": "estratosferica",
      "rss_feeds": [
        "https://www.dexerto.com/feed/",
        "https://www.gamespot.com/feeds/news/",
        "https://www.pcgamer.com/rss/"
      ],
      "max_per_feed": 3,
      "shuffle_articles": True,
      "max_ai_items": 15,
      "threads": {
        "user_id": "me",
        "state_key": "accounts/estratosferica/threads_state.json",
        "auto_post": true,
        "auto_post_limit": 1,
        "dry_run": false,
        "repost_enable": true,
        "repost_max_times": 3,
        "repost_window_days": 7
      },
      "r2": {
        "threads_media_prefix": "threads_media/estratosferica",
        "ig_queue_prefix": "ugc/ig_queue/estratosferica"
      }
    },
    {
      "account_id": "cliente2",
      "rss_feeds": [
        "https://www.dexerto.com/feed/"
      ],
      "max_per_feed": 2,
      "shuffle_articles": true,
      "max_ai_items": 10,
      "threads": {
        "user_id": "me",
        "state_key": "accounts/cliente2/threads_state.json",
        "auto_post": true,
        "auto_post_limit": 1,
        "dry_run": true,
        "repost_enable": true,
        "repost_max_times": 3,
        "repost_window_days": 7
      },
      "r2": {
        "threads_media_prefix": "threads_media/cliente2",
        "ig_queue_prefix": "ugc/ig_queue/cliente2"
      }
    }
  ]
}
