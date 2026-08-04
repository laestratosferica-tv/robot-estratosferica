# Cloudflare publisher scheduler

Disparador redundante del workflow `scheduled-meta-publisher.yml` cada cinco minutos.

- Consulta primero las ejecuciones recientes.
- No dispara si existe una ejecución en cola, activa o creada durante los últimos cuatro minutos.
- El token de GitHub se almacena únicamente como secreto cifrado del Worker.
- El workflow conserva su compuerta editorial: solo procesa piezas aprobadas y vencidas.

Despliegue: ejecutar manualmente `Deploy Cloudflare Publisher Scheduler` en GitHub Actions.
