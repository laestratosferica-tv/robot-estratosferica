# Radar de Jugadas v1

## Objetivo

Descubrir jugadas extraordinarias y creadores en Twitch y YouTube sin
descargar, copiar ni publicar sus videos.

El radar convierte metadatos de las plataformas en candidatos ordenados para
una revisión editorial y de derechos. Cada candidato conserva:

- plataforma, creador y enlace original;
- juego, título, fecha y visualizaciones;
- señales de competencia, emoción y actualidad;
- oportunidades comerciales posibles;
- estado explícito de derechos y próxima acción.

## Estado de seguridad

La versión 1 opera únicamente con metadatos:

- `reuse_allowed = false`;
- `download_allowed = false`;
- `republication_allowed = false`;
- `automatic_publish_allowed = false`;
- aprobación humana obligatoria;
- costo medido de USD 0.

El enlace puede usarse para curaduría, contacto con el creador o inserción
oficial cuando las condiciones de la plataforma lo permitan. La republicación
solo puede habilitarse con autorización verificable.

## Puntuación inicial

La prioridad combina:

1. visualizaciones;
2. frescura, con máximo 14 días;
3. señales como `ace`, `clutch`, `pentakill`, final, torneo o récord;
4. señales de conversación como reacción, increíble o imposible.

Este puntaje sirve para ordenar la revisión; no demuestra titularidad ni
permiso de uso.

## Conectores oficiales de solo lectura

El comando `python epic_play_radar.py --live` consulta:

- clips recientes por juego mediante Twitch Helix;
- búsquedas y estadísticas de videos mediante YouTube Data API.

El workflow `epic-play-radar-readonly.yml` solo puede iniciarse manualmente.
Toma las credenciales desde GitHub Secrets, genera una cola descargable y no
llama endpoints de publicación ni descarga de medios.

YouTube queda limitado inicialmente a dos búsquedas por ejecución
(aproximadamente 200 unidades de cuota de búsqueda). Twitch consulta seis
juegos prioritarios, incluidos EA Sports FC y Gran Turismo.

El módulo heredado que descarga videos no se reutiliza.
