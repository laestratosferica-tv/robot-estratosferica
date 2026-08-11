# PromoDetector: automatización y SEO v1

## Objetivo

Mantener un catálogo rastreable por Google sin publicar señales vencidas, inventadas o copiadas de un comercio.

## Ciclo automático

1. Descubrir candidatos desde fuentes oficiales o feeds autorizados.
2. Verificar identidad del producto, mercado, vigencia, URL HTTPS y evidencia.
3. Calificar sobre 10 con criterio editorial propio.
4. Publicar páginas estáticas rápidas con URL canónica, sitemap y datos estructurados.
5. Revalidar señales comerciales cada 24 horas.
6. Retirar del catálogo activo lo vencido; conservar la página editorial cuando siga aportando contexto.

La ejecución está programada a las 06:00, 12:00 y 18:00 de Colombia. El sistema es `fail-closed`: si no existe evidencia fresca, no habilita una señal comercial.

## SEO

- `robots.txt` permite el contenido público y bloquea superficies internas de revisión.
- `sitemap.xml` enumera únicamente páginas públicas canónicas.
- `feed.xml` expone novedades editoriales.
- La portada declara `WebSite` y `SearchAction` mediante JSON-LD.
- Las páginas de producto deben aportar valoración propia, utilidad, límites y comparación. Copiar descripciones del comercio está prohibido.
- Precio, descuento y disponibilidad solo se muestran con fuente y hora de observación vigentes.

## Configuración todavía necesaria

Para descubrir y actualizar productos sin intervención humana falta conectar una fuente comercial autorizada. Amazon Product Advertising API 5.0 no debe ampliarse porque Amazon dirige la migración a Creators API. La integración futura debe aportar, por mercado: credenciales de Creators API, associate tag, imagen autorizada, disponibilidad, precio, moneda y hora de observación.

Para medir posicionamiento falta registrar `https://promodetector.co/` en Google Search Console y enviar `https://promodetector.co/sitemap.xml`. El DNS ya contiene verificación del dominio, pero la propiedad y el envío deben comprobarse en Search Console.

## Regla de publicación

El trabajo programado construye y valida, pero no despliega automáticamente una página con señales comerciales bloqueadas. La publicación automática solo se habilita cuando la fuente autorizada está configurada y las pruebas reportan al menos una señal fresca.
