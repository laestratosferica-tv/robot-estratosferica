Run python -u main.py
RUNNING MEDIA ENGINE (Threads REAL + IG Queue + REEL AUTO + Multi-account via accounts.json)

===== RUN ACCOUNT: estratosferica =====
Obteniendo artículos (RSS)...
9 artículos candidatos tras mix/balance (MAX_PER_FEED=***, SHUFFLE=True)
FEEDS EN ESTA CORRIDA: ['https://www.dexerto.com/feed/', 'https://www.gamespot.com/feeds/news/', 'https://www.pcgamer.com/rss/']
TOTAL FEEDS: ***
STATE posted_items: 10
Seleccionado (NUEVO): https://www.dexerto.com/youtube/youtuber-accused-of-killing-pregnant-girlfriend-while-using-pre-recorded-gta-stream-as-alibi-***24171/
Publicando en Threads (NUEVO)...
IMAGE re-hosted on R2: https://pub-89***7244ee72549569***14507bb8f4***1e.r2.dev/threads_media/estratosferica/009e2***c9e042cab5.jpg
Container created: 17846884***8***691069
Threads publish response: {'id': '180962***7758***11466'}
Auto-post Threads: OK ✅
Generando REEL automático (***s)...
REEL: llamando generate_reel_mp4_bytes
REEL: falló (no rompe el run): Command '['ffmpeg', '-y', '-nostdin', '-hide_banner', '-loglevel', 'error', '-stream_loop', '-1', '-i', 'assets/bg.jpg', '-i', '/tmp/tmp9uikdif9/news.jpg', '-i', 'assets/logo.png', '-filter_complex', "[0:v]scale=1080:1920,format=rgba[bg];[1:v]scale=960:-1,format=rgba[news];[2:v]scale=700:-1,format=rgba[logo];[bg][news]overlay=(W-w)/2:520:format=auto[bg2];[bg2][logo]overlay=(W-w)/2:170:format=auto[bg***];[bg***]drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:text='YouTuber accused of killing pregnant girlfriend while using pre-recorded GTA stream as alibi':x=60:y=1***20:fontsize=48:fontcolor=white:box=1:boxcolor=black@0.45:boxborderw=24,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:text='Sigue para más hype gamer 🚀':x=60:y=***40:fontsize=42:fontcolor=white:box=1:boxcolor=black@0.***5:boxborderw=18[vout]", '-map', '[vout]', '-an', '-t', '***', '-r', '***0', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-shortest', '/tmp/tmpa7c***s1x/reel.mp4']' timed out after 120 seconds
IG queue guardado en R2: ugc/ig_queue/estratosferica/20260224_222057_8b85***9887***26.json
Archivo guardado en R2: accounts/estratosferica/runs/editorial_run_20260224_222057.json
RUN COMPLETED: estratosferica

===== RUN ACCOUNT: cliente2 =====
Obteniendo artículos (RSS)...
4 artículos candidatos tras mix/balance (MAX_PER_FEED=2, SHUFFLE=True)
FEEDS EN ESTA CORRIDA: ['https://www.dexerto.com/feed/', 'https://www.pcgamer.com/rss/']
TOTAL FEEDS: 2
STATE posted_items: 6
Seleccionado (NUEVO): https://www.pcgamer.com/games/reddit-fined-nearly-usd20-million-by-uk-online-privacy-regulator-for-using-childrens-data-unlawfully-potentially-exposing-them-to-inappropriate-and-harmful-content/
Publicando en Threads (NUEVO)...
[DRY_RUN] Threads post: Reddit ha sido multado con casi 20 millones de dólares por el regulador de privacidad en línea del Reino Unido, debido al uso indebido de datos de niños, lo que podría haberlos expuesto a contenido inapropiado y dañino. Este caso resalta la importancia de proteger a los usuarios más jóvenes en las
Fuente:https://www.pcgamer.com/games/reddit-fined-nearly-usd20-million-by-uk-online-privacy-regulator-for-using-childrens-data-unlawfully-potentially-exposing-them-to-inappropriate-and-harmful-content/
[DRY_RUN] Image source: https://cdn.mos.cms.futurecdn.net/kfJL***ERtgKibogoVEn2qjE-2560-80.jpg
Auto-post Threads: OK ✅
Generando REEL automático (***s)...
REEL: llamando generate_reel_mp4_bytes
REEL: falló (no rompe el run): ffmpeg falló:
STDERR:
[AVFilterGraph @ 0x55ba598c1700] No such filter: 'potentia:x=60:y=1***20:fontsize=48:fontcolor=white:box=1:boxcolor=black'
Failed to set value '[0:v]scale=1080:1920,format=rgba[bg];[1:v]scale=960:-1,format=rgba[news];[2:v]scale=700:-1,format=rgba[logo];[bg][news]overlay=(W-w)/2:520:format=auto[bg2];[bg2][logo]overlay=(W-w)/2:170:format=auto[bg***];[bg***]drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:text='Reddit fined nearly $20 million by UK online privacy regulator for \'using children’s data unlawfully, potentia':x=60:y=1***20:fontsize=48:fontcolor=white:box=1:boxcolor=black@0.45:boxborderw=24,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:text='Sigue para más hype gamer 🚀':x=60:y=***40:fontsize=42:fontcolor=white:box=1:boxcolor=black@0.***5:boxborderw=18[vout]' for option 'filter_complex': Filter not found
Error parsing global options: Filter not found

IG queue guardado en R2: ugc/ig_queue/cliente2/20260224_222109_f5***5fa240191.json
Archivo guardado en R2: accounts/cliente2/runs/editorial_run_20260224_222109.json
RUN COMPLETED: cliente2

===== SUMMARY =====
{
  "runs": [
    {
      "account_id": "estratosferica",
      "run_key": "accounts/estratosferica/runs/editorial_run_20260224_222057.json",
      "payload": {
        "generated_at": "2026-02-24T22:20:57.995597+00:00",
        "account_id": "estratosferica",
        "mix": {
          "shuffle": true,
          "max_per_feed": ***,
          "max_ai_items": ***
        },
        "settings": {
          "verify_news": false,
          "enable_trends": false,
          "enable_reels": true,
          "reel_seconds": ***
        },
        "result": {
          "posted_count": 1,
          "results": [
            {
              "link": "https://www.dexerto.com/youtube/youtuber-accused-of-killing-pregnant-girlfriend-while-using-pre-recorded-gta-stream-as-alibi-***24171/",
              "mode": "new",
              "posted": true,
              "threads": {
                "ok": true,
                "container": {
                  "id": "17846884***8***691069"
                },
                "publish": {
                  "id": "180962***7758***11466"
                },
                "image_url": "https://pub-89***7244ee72549569***14507bb8f4***1e.r2.dev/threads_media/estratosferica/009e2***c9e042cab5.jpg"
              }
            }
          ]
        }
      }
    },
    {
      "account_id": "cliente2",
      "run_key": "accounts/cliente2/runs/editorial_run_20260224_222109.json",
      "payload": {
        "generated_at": "2026-02-24T22:21:09.***20***9+00:00",
        "account_id": "cliente2",
        "mix": {
          "shuffle": true,
          "max_per_feed": 2,
          "max_ai_items": 10
        },
        "settings": {
          "verify_news": false,
          "enable_trends": false,
          "enable_reels": true,
          "reel_seconds": ***
        },
        "result": {
          "posted_count": 1,
          "results": [
            {
              "link": "https://www.pcgamer.com/games/reddit-fined-nearly-usd20-million-by-uk-online-privacy-regulator-for-using-childrens-data-unlawfully-potentially-exposing-them-to-inappropriate-and-harmful-content/",
              "mode": "new",
              "posted": true,
              "threads": {
                "ok": true,
                "dry_run": true
              }
            }
          ]
        }
      }
    }
  ]
}
