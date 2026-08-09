# Objetivo Amazon Influencer: seguimiento verificable

## Propósito

Medir semanalmente si `@laestratosfericatv` está construyendo audiencia,
actividad e interacción sostenibles antes de volver a solicitar Amazon
Influencer. Amazon no publica un umbral exacto; por eso el sistema no afirma
que su puntaje interno represente la decisión de Amazon.

## Qué mide

- seguidores y variación semanal;
- publicaciones nuevas durante la semana;
- alcance o vistas;
- interacciones y tasa de interacción;
- métricas adicionales que la API entregue para la cuenta.

El primer corte queda marcado como `partial`, porque todavía no existe una
línea base con la cual calcular crecimiento y frecuencia. A partir del segundo
corte se habilita la comparación semanal.

## Puntaje interno

El puntaje de 0 a 100 sirve únicamente para priorizar trabajo:

- audiencia: 30 %;
- interacción: 35 %;
- constancia: 20 %;
- crecimiento: 15 %.

Las metas iniciales son configurables y conservadoras: 2.000 seguidores, tres
publicaciones semanales, 3 % de interacción sobre alcance y 1 % de crecimiento
semanal. No son requisitos oficiales ni garantizan aceptación.

Una señal de 80 o más, con datos completos, cambia la recomendación a
`review_reapplication_manually`. Nunca presenta una solicitud ni cambia la
estrategia de forma automática.

## Sostenibilidad antes de volver a solicitar

Una semana buena no basta. El evaluador de reingreso exige cuatro cortes
semanales y revisa, con metas internas visibles:

- cuatro semanas completas;
- al menos tres publicaciones en cada semana;
- interacción de 3 % o más en al menos tres de cuatro semanas;
- crecimiento positivo de seguidores en al menos tres de cuatro semanas;
- meta interna de 2.000 seguidores.

Solo si todas las compuertas pasan genera
`ready_for_human_reapplication_review: true`. Esto sigue sin representar un
criterio oficial ni garantizar que Amazon acepte la cuenta.

## Ejecución

El workflow `instagram-account-growth-readonly.yml` admite ejecución manual y
requiere los secretos existentes `IG_ACCESS_TOKEN` e `IG_USER_ID`. La primera
versión queda deliberadamente sin cron para respetar la compuerta global que
mantiene desactivadas las ejecuciones programadas; la programación semanal se
activa después de validar el primer corte real.

Los archivos se entregan como artefactos privados por 90 días:

```text
artifacts/instagram-account-growth/YYYY-MM-DD.json
artifacts/instagram-account-growth/latest.json
artifacts/instagram-account-growth/reapplication-readiness.json
```

## Seguridad

- solo lectura;
- no publica;
- no solicita Amazon Influencer;
- no guarda tokens;
- oculta el token si Meta devuelve un error;
- cualquier recomendación exige revisión humana.
