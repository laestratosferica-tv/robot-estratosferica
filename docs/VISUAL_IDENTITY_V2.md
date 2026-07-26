# Identidad visual adaptativa V2

## Principio

**Reconocible siempre, repetitiva nunca.**

La identidad se construye con una mezcla protegida:

- 70% ADN fijo de La Estratosférica.
- 20% lenguaje visual de la categoría.
- 10% señal de temporada o tendencia.

## ADN fijo

- Arco orbital.
- Sello corto `LETV`.
- Degradado cian → violeta → magenta.
- Base oscura premium.
- Firma editorial de La Estratosférica.

Estos elementos deben permanecer incluso cuando cambien la fotografía, el color
de acento, la composición, la temperatura o la textura.

## Lenguaje por categoría

| Pilar | Dirección |
|---|---|
| Gaming | Cinética, eléctrica y de alto contraste |
| Tecnología e IA | Precisa, fría y cromática |
| Publicidad | Editorial, lúdica y de campaña |
| Moda | Editorial, limpia y con mayor aire |
| Gastronomía | Cálida, sensorial y cercana |
| Estilo de vida | Abierta, natural y ligera |
| Lujo | Espaciosa, sobria y con brillo controlado |
| Negocios | Clara, optimista y basada en señales |

La categoría debe cambiar más que el color: también modifica contraste,
saturación, escala del titular, ritmo, textura y composición.

## Tendencias y temporadas

Las tendencias nunca se adoptan automáticamente. Cada perfil debe:

1. Tener nombre y vigencia.
2. Ocupar como máximo el 10% de la expresión visual.
3. Ser revisado antes de activarse en `accounts.json`.
4. Volver a `evergreen` si el perfil no existe o no ha sido aprobado.

Perfil inicial: `sport_luxe_2026_q3`, vigente de julio a septiembre de 2026.

## Seguridad

- Este sistema no enciende publicaciones ni cambia credenciales.
- Añadir una tendencia no autoriza su uso automático.
- Las señales no aprobadas caen al perfil `evergreen`.
- Para revertir, definir `active_trend_profile` como `evergreen`.
