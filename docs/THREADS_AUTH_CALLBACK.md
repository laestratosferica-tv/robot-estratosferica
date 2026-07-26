# Retorno seguro de autorización de Threads

## URL vigente

`https://estratosferica-threads-auth.joseluca83.chatgpt.site`

Esta dirección reemplaza el retorno histórico de Railway:

`https://robot-estratosferica-production.up.railway.app/callback/`

## Responsabilidades

La página:

- recibe el código temporal que Meta añade a la URL;
- lo muestra únicamente en el navegador;
- permite copiarlo para completar el intercambio seguro;
- no contiene la clave secreta de la aplicación;
- no almacena ni transmite el código;
- no crea ni publica contenido.

El robot editorial y los publicadores continúan ejecutándose en GitHub Actions.
La página es solo el punto de retorno exigido por OAuth.

## Seguridad operativa

- `PRODUCTION_ARMED` permanece cerrado.
- La renovación del token no habilita publicaciones por sí sola.
- El código de autorización no debe copiarse en issues, commits, logs ni conversaciones.
- Las credenciales finales deben mantenerse en GitHub Actions Secrets.
