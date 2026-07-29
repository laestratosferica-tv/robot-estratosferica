# Portafolio de fuentes y monetización

## Criterio editorial

La Estratosférica no se convierte en un medio generalista. Gaming, esports y
cultura digital siguen siendo el núcleo. Tecnología, gastronomía y lujo entran
solo cuando tienen una relación demostrable con jugadores, comunidades,
creadores, eventos, entretenimiento o marcas que quieran llegar a ese público.

## Pilares de contenido

El radar no se limita a noticias. Debe alimentar siete productos editoriales:

| Pilar | Ejemplos | Fuente prioritaria |
| --- | --- | --- |
| Jugada épica | clutch, remontada, récord, gol o final inesperado | Twitch/YouTube con permiso o inserción |
| Competencia | resultados, tabla, ranking, posiciones y calendario | organizador, liga o API oficial |
| Protagonistas | mejores jugadores, equipos, revelaciones y perfiles | liga/equipo + datos verificables |
| Lanzamientos | juego, temporada, expansión, demo y actualización | publisher o estudio |
| Utilidad | códigos, claves, recompensas y drops | publisher/plataforma; nunca filtraciones |
| Industria | patrocinios, audiencias, negocios y tendencias | fuente primaria + prensa especializada |
| Cultura gamer | hardware, comida, moda, lujo, eventos y experiencias | marca + fuente especializada |

FC / EA Sports FC funciona como juego bandera, sin excluir Valorant, League of
Legends, CS2, Fortnite, Free Fire, Mobile Legends, F1, Gran Turismo, Roblox,
Minecraft y otros juegos con tracción regional.

El radar separa las fuentes en tres rutas:

1. **Automática primaria:** se puede usar como origen verificable de una noticia.
2. **Automática con corroboración:** descubre tendencias, pero la afirmación
   central debe contrastarse con una fuente primaria.
3. **Radar comercial/manual:** identifica colaboraciones, anunciantes,
   afiliados, eventos o temas especiales; no alimenta el flujo diario.

## Fuentes automáticas actuales

| Fuente | Rol | Valor editorial | Oportunidad comercial |
| --- | --- | --- | --- |
| Xbox Wire ES-LATAM | Primaria | Xbox, Game Pass, PC, cloud y ecosistema | hardware, suscripciones, accesorios |
| Google Blog Latinoamérica | Primaria | IA, plataformas e innovación regional | software, formación, servicios B2B |

## Candidatas a prueba en vivo

Estas rutas existen públicamente, pero no se activan hasta medir relevancia,
duplicados, calidad de fecha y porcentaje de ruido.

| Fuente | Ruta candidata | Filtro obligatorio | Monetización |
| --- | --- | --- | --- |
| WIRED Cultura | `https://es.wired.com/feed/cultura/rss` | gaming, streaming, creadores, cultura digital | entretenimiento y colaboraciones |
| WIRED Gadgets | `https://es.wired.com/feed/gadgets/rss` | hardware, audio, pantallas, PC, móvil gamer | afiliados y guías de compra |
| Xataka | `https://www.xataka.com/index.xml` | gaming, IA, PC, consolas, periféricos | afiliados y patrocinio tecnológico |

Para pasar a automático, cada fuente debe:

- entregar fecha legible por máquina;
- conservar el enlace canónico;
- superar 60 % de candidatos relacionados con el foco editorial después del
  filtro temático;
- no aportar más de 25 % de la cola diaria;
- permanecer marcada como fuente secundaria y exigir corroboración;
- usar solo título, resumen y enlace; nunca reutilizar medios sin licencia.

## Gaming y esports

| Fuente | Ruta | Uso |
| --- | --- | --- |
| Twitch Helix | Primaria/API | descubrir clips, vistas, juego, creador y URL canónica |
| YouTube Data API | Primaria/API | lanzamientos, directos, canales oficiales y videos autorizados |
| Riot Games / LoL Esports | Primaria | competencias, lanzamientos y ecosistema |
| gamescom latam | Primaria/manual | eventos, prensa, marcas y oportunidades regionales |
| Esports Insider | Especialista | industria, patrocinios, derechos y negocios |
| SportsPro | Especialista | convergencia deporte, gaming y medios |
| GDC / reportes de industria | Investigación | empleo, IA, desarrollo y economía del sector |
| BCG Gaming | Investigación | modelos de negocio y señales de monetización |

### Recuperación segura del sistema heredado

El repositorio conserva motores históricos:

- `Mode E` y `Mode F`: búsqueda y puntuación de clips de Twitch.
- `Mode Y`: búsqueda y descarga de videos de YouTube.
- `Mode Z`: radar de conversación gamer en Reddit.

La nueva arquitectura puede reutilizar metadatos, filtros, puntuación,
anti-duplicado y atribución. No debe reactivar la descarga/republicación de
videos ajenos sin validar derechos.

| Uso de video | Estado |
| --- | --- |
| Enlazar al clip o video original con atribución | Permitido para curaduría |
| Insertar el reproductor oficial de Twitch/YouTube | Preferido para sitio web |
| Solicitar clip a creador mediante convocatoria y autorización | Permitido con registro de licencia |
| Usar tráiler o press kit con términos claros | Permitido según licencia |
| Remix habilitado por la propia plataforma | Permitido dentro de sus reglas |
| Descargar, recortar y republicar un video ajeno | Bloqueado sin permiso explícito |
| Extraer “claves” filtradas, hacks o contenido engañoso | Bloqueado |

### Escalera creativa de derechos

La falta del recurso ideal no detiene automáticamente una historia. El sistema
debe recorrer estas alternativas, en orden, hasta encontrar una solución
publicable:

1. captura propia del juego bajo las reglas vigentes del editor;
2. tráiler, press kit o banco oficial con permiso aplicable;
3. material Creative Commons o de stock compatible con uso comercial;
4. permiso escrito del creador del gameplay, guardando alcance y atribución;
5. herramienta de remix o inserción ofrecida por la propia plataforma;
6. gameplay de una entrega anterior del mismo universo, claramente presentado
   como archivo o apoyo visual y sin atribuirle funciones del juego nuevo;
7. recreación visual original, ilustración, animación, interfaz o 3D inspirados
   en el tema sin copiar activos protegidos;
8. fragmento mínimo para crítica, análisis o noticia solo después de revisión
   editorial y jurídica del caso concreto.

Ningún informe de bloqueo puede terminar únicamente con “no se puede”. Debe
registrar las rutas evaluadas, ofrecer alternativas concretas y recomendar la
mejor solución. Mientras se confirma una licencia, la publicación permanece
desactivada, pero el trabajo editorial y el prototipo continúan con material
seguro o sustituto.

El sistema puede recortar, reencuadrar, subtitular, narrar, sonorizar y adaptar
material autorizado para convertirlo en una pieza propia. Estas
transformaciones mejoran la creatividad, pero no sustituyen la licencia.

Cuando el gameplay provenga de otra persona existen dos capas de derechos: el
contenido del juego y la grabación o edición del creador. La autorización del
editor no elimina la necesidad de permiso del creador, salvo que la licencia
del video permita expresamente la reutilización.

Nunca se ocultará el origen para que “no se note”. La edición puede integrar
visualmente material de distintas fuentes, pero debe evitar afirmar o insinuar
que imágenes de una versión anterior pertenecen a la versión nueva.

El primer producto recuperable es un **Radar de Jugadas**: descubre el clip,
calcula relevancia, conserva creador/juego/URL/vistas y prepara una ficha para
revisión. La publicación automática del archivo de video queda separada y
desactivada.

## Tecnología e IA

| Fuente | Ruta | Uso |
| --- | --- | --- |
| WIRED en Español | Candidata RSS | cultura digital, IA y dispositivos |
| Xataka | Candidata RSS | consumo tecnológico y guías de compra |
| Think with Google | Investigación | comportamiento de audiencia y publicidad |
| NVIDIA, AMD, Intel y fabricantes | Primaria/manual | hardware, IA, PC y anuncios verificables |
| PlayStation, Nintendo, Epic y Steam | Primaria/manual | plataformas, juegos, tiendas y ecosistemas |

## Gastronomía relacionada con el target

No se publican recetas o noticias gastronómicas generales. Solo entran:

- alimentos y bebidas creados para gamers o vinculados a franquicias;
- activaciones de restaurantes, delivery o snacks durante eventos;
- experiencias gastronómicas en estadios, festivales y arenas;
- tecnología para restaurantes y entretenimiento;
- tendencias útiles para alianzas comerciales.

Fuentes de radar: Fine Dining Lovers LATAM, Directo al Paladar, The Food Tech,
James Beard Foundation y comunicados oficiales de marcas. Permanecen en modo
manual hasta tener filtros de cruce con gaming, eventos o tecnología.

## Lujo y lifestyle relacionado con el target

Solo entra lujo conectado con cultura digital:

- setups, audio, pantallas, sillas y computadores premium;
- moda y relojería en colaboraciones con juegos o esports;
- hoteles, viajes y hospitalidad alrededor de eventos;
- vehículos, simuladores, coleccionables y experiencias;
- acuerdos entre marcas premium, creadores y franquicias.

Fuentes de radar: GQ México y Latinoamérica, Robb Report, Hypebeast, Luxury
Lifestyle Magazine y comunicados oficiales de las marcas. La fuente secundaria
sirve para descubrir; la marca o el organizador debe confirmar el dato.

## Producto editorial monetizable

| Producto | Señal de fuente | Ingreso posible |
| --- | --- | --- |
| Guía “vale la pena” | lanzamiento + precio + disponibilidad regional | afiliado |
| Comparativa de setup | varios fabricantes y pruebas verificables | afiliado + patrocinio |
| Radar de eventos LATAM | organizadores y publishers | cobertura patrocinada |
| Colaboración gamer de la semana | marca + franquicia | branded content |
| Dónde ver/jugar/comer | evento + comercio local | referido local |
| Informe de tendencia | industria + comportamiento de audiencia | patrocinio B2B |

## Regla de independencia

Un enlace afiliado, regalo, invitación o patrocinio nunca cambia el veredicto
editorial. Toda relación comercial debe declararse. La automatización puede
detectar una oportunidad, pero no puede insertar enlaces afiliados ni publicar
contenido patrocinado sin revisión y aprobación humana.
