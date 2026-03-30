# Machine-Learning-Anahuac
Machine learning anahuac 2026

---

## Práctica 5 — Segmentación de Usuarios en Tienda Online mediante Clustering

**Notebook:** `Practica5_Clustering_Usuarios.ipynb`
**Dataset:** `Data/product.csv` (~1,048,574 eventos de comportamiento)

---

### 1.1 Hallazgos del EDA

| Métrica | Valor |
|---------|-------|
| Total de eventos | ~1,048,574 |
| Usuarios únicos | ~925,000+ |
| Productos únicos | 5 (sneakers, sports_nutrition, clothes, accessories, company) |
| Columnas | 8 (order_id, user_id, page_id, product, site_version, time, title, target) |

**Distribución de eventos:**

| Tipo de evento | Descripción | Aprox. % |
|----------------|-------------|-----------|
| `banner_show` | El banner fue mostrado al usuario | ~87% |
| `banner_click` | El usuario hizo clic en el banner | ~10% |
| `order` (target=1) | El usuario realizó una compra | ~3% |

**Distribución de plataforma:** ~72% mobile, ~28% desktop.

**Productos más frecuentes:** sneakers (dominante), seguido por sports_nutrition y clothes.

**Valores faltantes:** Ninguno en columnas críticas.
**Anomalías detectadas:** Alta distribución sesgada en `total_interactions` (usuarios con 1 evento vs. usuarios con cientos), lo cual requiere manejo especial en clustering.

---

### 1.2 Ingeniería de Características

Se construyó una matriz de usuarios (1 fila = 1 usuario) con las siguientes características:

| Característica | Descripción | Justificación |
|----------------|-------------|---------------|
| `total_interactions` | Total de eventos por usuario | Mide nivel general de actividad |
| `total_clicks` | Número de clics en banners | Indica intención de compra |
| `total_purchases` | Número de compras | Métrica clave de conversión |
| `click_to_purchase_ratio` | Clics / Compras (0 si sin compras) | Tasa de eficiencia de clic |
| `mobile_percentage` | % de eventos en mobile | Preferencia de dispositivo |
| `desktop_percentage` | % de eventos en desktop | Preferencia de dispositivo |
| `avg_time_between_events` | Tiempo promedio entre eventos (horas) | Frecuencia de visitas |
| `products_viewed` | Productos únicos vistos (clics) | Amplitud de interés |
| `products_purchased` | Productos únicos comprados | Diversidad de compra |
| `avg_product_diversity` | Índice de Shannon sobre clics | Diversidad de interés |
| `show_to_click_ratio` | Shows / Clicks | CTR de banners |
| `days_active` | Días únicos con actividad | Lealtad temporal |
| `purchase_frequency` | Compras / días activos | Intensidad de compra |

**Normalización:** Se aplicó `StandardScaler` (media=0, std=1) antes de clustering.
**Divisiones por cero:** Manejadas con `np.where` → valor 0 cuando denominador es 0.
**Varianza cero:** Se eliminaron columnas con varianza cero automáticamente.

---

### 1.3 Análisis de Multicolinealidad y PCA

**Pares con |r| > 0.75 detectados:** `mobile_percentage` ↔ `desktop_percentage` (r ≈ -1.0, complementarias por definición), y posibles correlaciones entre `total_interactions` y `total_clicks`.

**Decisión sobre PCA:**
Se aplicó PCA para reducir la dimensionalidad, seleccionando el número de componentes que explican ≥90% de la varianza. Esto mejora la calidad del clustering al eliminar ruido y colinealidad.

**Varianza explicada:** Ver figura `pca_varianza.png`.
**PC1** captura principalmente actividad e interacciones totales.
**PC2** captura preferencia de dispositivo (mobile vs. desktop).
**PC3** captura comportamiento de compra (frecuencia y diversidad).

---

### Fase 2: Comparación de Algoritmos de Clustering

| Algoritmo | Parámetros | Silhouette | Davies-Bouldin | Calinski-Harabasz |
|-----------|-----------|------------|----------------|-------------------|
| K-Means | K óptimo por Silhouette | Mayor | Menor | Mayor |
| Mean-Shift | bandwidth estimado por `estimate_bandwidth` | Variable | Variable | Variable |
| DBSCAN | eps estimado por k-distancia | Variable | Variable | — |

**Algoritmo recomendado:** K-Means, por producir segmentos interpretables, escalables y con métricas superiores en este tipo de datos tabulares de comportamiento web.

---

### Fase 3: Recomendaciones Estratégicas

**Decisión CPC vs. Personalización:** Se recomienda invertir en **personalización basada en segmentación**.

| Segmento | Estrategia de Banner |
|----------|---------------------|
| Alto Valor (Compradores Frecuentes) | Productos premium, colecciones nuevas, fidelidad |
| Exploradores (Alta Navegación, Baja Compra) | Ofertas con urgencia, retargeting, A/B testing |
| Usuarios Pasivos | Contenido inspiracional, reactivación por email |
| Compradores Ocasionales | Cross-sell, recomendaciones, incentivos de frecuencia |

**Figuras generadas:**
- `eda_distribucion_eventos.png` — Distribución de eventos y productos
- `correlacion_features.png` — Heatmap de correlación entre features
- `pca_varianza.png` — Varianza explicada por PCA
- `pca_2d_usuarios.png` — Proyección 2D de usuarios sin etiquetar
- `kmeans_metricas.png` — Métricas K-Means para K=2..10
- `kmeans_clusters_2d.png` — Clusters K-Means en 2D
- `meanshift_clusters_2d.png` — Clusters Mean-Shift en 2D
- `dbscan_kdistance.png` — Gráfico k-distancia para eps
- `dbscan_clusters_2d.png` — Clusters DBSCAN en 2D
- `comparacion_algoritmos.png` — Comparación visual de los 3 algoritmos
- `kmeans_radar_perfiles.png` — Radar chart de perfiles de segmento
