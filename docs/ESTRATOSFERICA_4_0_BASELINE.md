# Estratosférica 4.0 — Base y evolución editorial

## Decisión de arquitectura

Estratosférica 4.0 nace desde la base funcional de Estratosférica 3.0, pero se
desarrolla en la rama `estratosferica-4.0`. La rama `main` conserva la versión
3.0 estable. Ningún experimento editorial de 4.0 modifica producción hasta que
sea validado y aprobado.

## Qué se hereda de 3.0

- Integraciones, credenciales externas y publicación en nube.
- Programación multiplataforma y Google Calendar.
- Avatares, voces, looks e identidades aprobadas de Joseverso, Nova y Rami.
- QA, seguridad, fuentes, derechos, métricas e historial editorial.
- HeyGen, Wan, HyperFrames/FFmpeg y la infraestructura de renderizado.

Los secretos no se copian a archivos. Se consumen mediante las variables y
entornos protegidos existentes.

## Qué cambia en 4.0

El objetivo deja de ser solamente producir piezas correctas. Cada short debe
resolver una necesidad real y superar tres compuertas:

1. **Gancho:** detiene el scroll y formula una promesa concreta.
2. **Tensión:** sostiene interés mediante progresión, consecuencias o una
   brecha de curiosidad legítima.
3. **Recompensa:** paga exactamente la promesa con utilidad, prueba, giro,
   emoción o aprendizaje.

Una pieza visualmente buena que no supere estas tres compuertas no entra en
producción.

## Flujo de trabajo

`necesidad → promesa → tres ganchos → tensión → recompensa → guion técnico → producción → QA → aprobación → nube → métricas → aprendizaje`

## Separación operativa

| Sistema | Función |
|---|---|
| Estratosférica 3.0 / `main` | Base estable y continuidad de producción |
| Estratosférica 4.0 / `estratosferica-4.0` | Investigación y evolución de shorts |
| Producción pública | Solo recibe piezas aprobadas y cambios verificados |

## Primera fase de 4.0

1. Crear una rúbrica ejecutable de Gancho, Tensión y Recompensa.
2. Auditar piezas anteriores y establecer línea base.
3. Producir un piloto por personaje sin cambiar identidades.
4. Probar una sola variable por versión.
5. Comparar retención inicial, duración media, finalización, repeticiones,
   guardados, compartidos, comentarios y seguidores.
6. Convertir patrones repetidos en reglas de producción.

## Regla de promoción

Los aprendizajes pasan a `main` únicamente cuando tengan evidencia suficiente,
pruebas técnicas aprobadas y autorización expresa para fusionar o publicar.
