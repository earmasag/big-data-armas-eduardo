# Proyecto Final — Gerencia y Almacenamiento en Big Data
## Pronóstico Hidrológico Multi-Estación del Río Orinoco a +14 Días con CRISP-DM, Machine Learning, Almacenamiento Híbrido (Parquet + MongoDB) y Dashboard Contenedorizado

- **Institución:** Fundación Universitaria Los Libertadores
- **Curso:** Gerencia y Almacenamiento en Big Data
- **Modalidad:** Individual
- **Autor:** Eduardo Armas
- **Dataset:** Niveles Hidrométricos Diarios del Río Orinoco (4 estaciones en cascada: *Puerto Ayacucho*, *Caicara del Orinoco*, *Ciudad Bolívar* y *Palúa*, 2009–2024) integrados con Precipitación Diaria Satelital **NASA POWER (MERRA-2 `PRECTOTCORR`)** en 15 puntos de la cuenca (1981–2025) y el Índice Climático **NOAA CPC Oceanic Niño Index (`ONI`)** (**271.196 registros crudos integrados**).

---

## 1. Problema de Negocio y Selección del Dataset

### 1.1 Contexto y Pregunta Principal de Negocio
El río Orinoco posee una cuenca de $989.000\text{ km}^2$ y experimenta oscilaciones anuales de más de **16 metros** en la angostura de Ciudad Bolívar y más de **11 metros** en el puerto multimodal de Palúa (San Félix). Anticipar el nivel del río con dos semanas de antelación es crítico tanto para la **prevención de inundaciones urbanas** (como la crecida histórica de agosto de 2018, que alcanzó $18,34\text{ m s.n.m.}$ en Ciudad Bolívar) como para la **planificación del calado de buques comerciales** durante el estiaje (enero–marzo).

- **Pregunta de Negocio:** *¿Con qué precisión podemos anticipar el nivel diario del río Orinoco con **14 días de anticipación** ($t+14$) en las estaciones estratégicas de navegación y control de inundaciones (Ciudad Bolívar y Palúa) integrando la propagación aguas arriba (Puerto Ayacucho y Caicara), la precipitación satelital en la cuenca (NASA POWER) y el estado del fenómeno El Niño/La Niña (NOAA ONI)?*
- **Preguntas Específicas del Análisis:**
  1. *¿Cuánto tarda la onda de crecida desde las estaciones aguas arriba (Puerto Ayacucho y Caicara) en reflejarse en Ciudad Bolívar y Palúa, y cómo varía la capacidad relativa del río según la época hidrológica y la fase ENSO (La Niña, Neutro, El Niño)?*
  2. *¿En qué porcentaje reducen el error de pronóstico a 14 días (MAE y RMSE) los modelos de Machine Learning supervisados (`Ridge Regression`, `Random Forest` y `XGBoost Regressor`) frente a una Línea Base de persistencia (`Naive`, que asume $\hat{y}_{t+14} = y_t$)?*
  3. *¿Cuáles son las variables predictoras de mayor impacto en lenguaje de negocio y qué umbrales operativos permiten emitir alertas tempranas hidrologísticas?*

### 1.2 Fuentes Oficiales de Datos y Justificación Escrita del Volumen
| Fuente de Datos | Origen Oficial / URL | Período | Registros Crudos | Rol en el Pipeline |
|---|---|---|---:|---|
| **Niveles Limnimétricos del Río Orinoco** (`DATOS RIO ORINOCO.xlsx` / `dataset_combinado.csv`) |Registros Hidrométricos INAMEH / Repositorio Académico UCAB (`https://github.com/Cesar2926/Proyecto-IA-Orinoco/tree/main/data/raw`) | `2009-01-01` a `2025-03-31` (Ventana oficial: `2009–2024`) | `23.736` celdas diarias (`23.376` válidas en `2009–2024`; `74.732` en histórico desde 1974) | Variable objetivo $y_{t+14}$ y predictores autorregresivos / aguas arriba |
| **Precipitación Diaria NASA POWER** (15 estaciones en la cuenca) | NASA Langley Research Center POWER API (`https://power.larc.nasa.gov/api/temporal/daily/point`, parámetro `PRECTOTCORR`) | `1981-01-01` a `2025-12-31` | `246.540` registros diarios (`16.436` días $\times$ 15 estaciones) | Aporte pluvial local y agregado por subcuencas del Orinoco y Caroní |
| **Índice Climático NOAA CPC ONI** (`noaa_oni.ascii.txt`) | NOAA Climate Prediction Center (`https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt`) | `1950` a `2025` | `920` trimestres móviles | Clasificación macroclimática ENSO (*La Niña*, *Neutro*, *El Niño*) |
| **Total Integrado en el ETL** | **3 fuentes públicas oficiales** | **1950–2025 (Corte modelo: 2009–2024)** | **271.196 registros crudos** (`322.192` con histórico 1974) | **23.376 registros diarios limpios $\times$ 56 columnas en Parquet** |

> **Justificación formal del volumen de datos (< 500.000 registros):**
> De acuerdo con las condiciones de la guía (*"Volumen: preferiblemente más de 500.000 registros. Si el dataset elegido tiene menos, debe justificarse por escrito"*), se deja constancia de que en hidrología de grandes cuencas fluviales las estaciones limnimétricas oficiales y los productos satelitales diarios operan con **frecuencia de 1 observación por día** (`freq = D`). Aun integrando **45 años completos de precipitación satelital diaria en 15 puntos geográficos de la cuenca (246.540 registros)** junto con los niveles diarios de las 4 estaciones del río Orinoco y el índice NOAA ONI (**271.196 registros crudos totales**), la resolución física diaria del fenómeno acota el conteo de filas. Incrementar el número de filas por encima de 500.000 exigiría desagregar o interpolar artificialmente niveles a frecuencia horaria, lo cual vulneraría la regla del curso que **prohíbe estrictamente el uso de datos sintéticos**.

---

## 2. Integración con las Actividades del Curso

| Actividad / Unidad | Cómo se evidencia en este Proyecto Final |
|---|---|
| **Actividad 1: Arquitectura híbrida SQL/NoSQL** | Almacenamiento tabular analítico en **Apache Parquet** (`data/processed/orinoco_limpio.parquet`) combinado con documento rico JSON/BSON en **MongoDB** (`orinoco_bigdata.analisis_crispdm`) con metadatos, calidad, estadísticas, agregaciones, métricas de ML, 6 insights y 4 recomendaciones, más 3 consultas NoSQL en el notebook. |
| **Actividad 2: OLAP** | Construcción de **2 cubos multidimensionales OLAP** (`Año × Estación × Época Hidrológica` y `Fase ENSO × Estación × Trimestre`) persistidos en Parquet/MongoDB y conectados como filtros *slice, dice & drill-down* en el dashboard. |
| **Actividades 3 y 6: CRISP-DM y ETL** | Diagnóstico previo con tabla de 6 problemas de calidad y aplicación de las **4 técnicas ETL** (*Limpieza*, *Normalización*, *Agregación* y *Enriquecimiento*) con conteos exactos antes/después (de 34 anomalías/huecos a 0 nulos) y más de 10 variables derivadas. |
| **Actividad 4: Spark vs. pandas** | Benchmark empírico propio que mide filas, consumo de RAM antes/después de optimizar tipos (`-60,0%` RAM) y tiempo de ejecución (`1,73 s`), justificando técnicamente `pandas + PyArrow` frente al *overhead* JVM de Spark. |
| **Actividad 5: Entornos y kernels** | Entorno 100% reproducible con `requirements.txt`, variables de entorno en `.env.example` y contenedorización multi-servicio en **Docker Compose**. |
| **Actividad 7: Visualización** | **6 visualizaciones** diseñadas con la paleta accesible *Okabe-Ito* (apta para daltónicos), alto *data-ink ratio*, títulos con *insight* cuantitativo y ejes con unidades en español. |
| **Nuevo: Machine Learning (CRISP-DM 4 y 5)** | Evaluación libre de *data leakage* (embargo temporal de 14 días) comparando **Línea Base (`Naive`)** contra **`Ridge Regression`**, **`Random Forest`** y **`XGBoost Regressor`**, reduciendo el MAE en **53,01%** globales (**57,02%** en Ciudad Bolívar) con $R^2 = 0,9989$ y exportando `models/modelo_orinoco.joblib`. |
| **Nuevo: Dashboard Streamlit + PDF + Docker** | Reporte PDF generado automáticamente desde MongoDB (`reports/reporte_bigdata_orinoco.pdf`), aplicación Streamlit de 5 pestañas (`app/app.py`) y despliegue con un solo comando `docker compose up --build`. |

---

## 3. Arquitectura de la Solución y Decisión `pandas` vs. `Spark`

```mermaid
flowchart LR
    subgraph Fuentes["Datos Abiertos Oficiales (271.196 registros)"]
        R1["Niveles Río Orinoco (XLSX/CSV)"]
        R2["NASA POWER 15 Estaciones (CSV)"]
        R3["NOAA CPC ONI (ASCII)"]
    end

    subgraph Docker["docker compose up --build (Red: bigdata_net)"]
        P["Servicio 1: procesamiento\n(Python 3.12 + pandas + PyArrow + scikit-learn/XGBoost + ReportLab + Jupyter :8888)"]
        M[("Servicio 2: mongodb\n(MongoDB 7.0 :27017)\norinoco_bigdata.analisis_crispdm")]
        A["Servicio 3: app\n(Streamlit Dashboard :8501)\n5 Pestañas + Predicción en Vivo + Descarga PDF"]
        V[("Volúmenes Compartidos\ndata/raw · data/processed · models/ · reports/")]
    end

    Fuentes -->|"src/download_data.py"| V
    P -->|"Lee data/raw y escribe Parquet, .joblib y PDF"| V
    P -->|"Inserta documento JSON y ejecuta 3 consultas"| M
    V -->|"Solo lectura (sin reentrenar)"| A
    M -->|"Lee insights, métricas y metadatos"| A
```

### Justificación cuantitativa: `pandas` + `PyArrow` frente a `Spark`
- **Registros procesados:** `271.196` registros crudos integrados (`246.540` de NASA POWER + `23.736` de niveles + `920` de NOAA ONI).
- **Tiempo medido del ETL completo (4 técnicas + 2 cubos OLAP):** `1,73 segundos`.
- **Optimización de memoria RAM:** El DataFrame enriquecido de `23.376 filas × 56 columnas` pasa de **`10,98 MB` a `4,39 MB` en RAM** (**reducción del `60,0%`**) al convertir flotantes a `float32` y dimensiones de texto a `category`, quedando almacenado en disco en un archivo Parquet columnar de apenas **`1,53 MB`**.
- **Por qué no usar Spark en este volumen:** Levantar una `SparkSession` en la JVM consume más de `600 MB` de memoria base y añade `4–8 segundos` solo en inicialización y serialización Python $\leftrightarrow$ JVM. De acuerdo con la regla de arquitectura de la Actividad 4, cuando el dataset cabe en menos del 5% de la RAM de un nodo y el ETL vectorizado en `pandas` + `PyArrow` finaliza en menos de 5 segundos, una arquitectura mononodo optimizada es técnicamente superior, más rápida y más reproducible en contenedores Docker.

---

## 4. Estructura del Proyecto

```text
Armas_Eduardo_BigData_ProyectoFinal/
├── README.md                                              # Documentación completa, arquitectura, puertos y diagnóstico Docker
├── docker-compose.yml                                     # Orquestación de los 3 servicios: mongodb, procesamiento y app
├── .env.example                                           # Plantilla de variables de entorno y credenciales de MongoDB
├── requirements.txt                                       # Dependencias Python fijadas para reproducibilidad
├── data/
│   ├── raw/                                               # Se puebla automáticamente con python -m src.download_data
│   └── processed/                                         # Parquet limpio (orinoco_limpio.parquet) y cubos OLAP
├── notebooks/
│   └── Armas_Eduardo_BigData_ProyectoFinal.ipynb         # Notebook ejecutado completo sin errores con todas las salidas
├── src/
│   ├── __init__.py
│   ├── download_data.py                                   # Script de descarga/ingesta en data/raw/ con rutas relativas
│   ├── etl_pipeline.py                                    # ETL CRISP-DM (4 técnicas), cubos OLAP y 6 visualizaciones
│   ├── ml_pipeline.py                                     # Entrenamiento sin leakage (Naive, Ridge, RF, XGBoost) -> .joblib
│   ├── mongodb_store.py                                   # Persistencia y 3 consultas NoSQL en MongoDB + cierre de conexión
│   ├── generate_pdf.py                                    # Generación automática del reporte PDF desde MongoDB con ReportLab
│   ├── build_notebook.py                                  # Constructor y ejecutor limpio (Restart & Run All) del notebook
│   └── run_all.py                                         # Orquestador de punta a punta del pipeline
├── models/
│   └── modelo_orinoco.joblib                              # Modelo ganador entrenado listo para inferencia en el dashboard
├── reports/
│   ├── figures/                                           # Las 6 visualizaciones PNG de alta resolución (Actividad 7)
│   └── reporte_bigdata_orinoco.pdf                        # Reporte PDF generado automáticamente desde MongoDB
└── app/
    ├── Dockerfile                                         # Imagen Docker Python 3.12 para procesamiento y Streamlit
    └── app.py                                             # Dashboard Streamlit con las 5 secciones obligatorias
```

---

## 5. Instrucciones de Ejecución y Verificación (`Docker Compose` y Local)

### 5.0 Requisitos Previos
1. Tener instalados **Docker Desktop** (motor Compose v2) y, opcionalmente, **Python 3.12 + `uv`** para ejecución local.
2. **Credenciales de MongoDB:** el proyecto usa credenciales de ejemplo `changeme / changeme` en `.env.example` (copiar a `.env` con `cp .env.example .env` o `Copy-Item .env.example .env -Force` en Windows). **Cambie `changeme` por sus propias credenciales seguras antes de exponer el servicio**; no hay secretos reales versionados en el repositorio.
3. **Reinicio limpio del entorno:** si cambia credenciales o desea partir de un estado limpio (volumen `mongo_data` incluido), ejecute:
   ```bash
   docker compose down -v
   ```
   Esto detiene los contenedores y **elimina el volumen de datos de MongoDB**, evitando conflictos de usuarios con credenciales antiguas al volver a levantar la base con `.env` nuevo.

### 5.1 Ejecución con un solo comando en Docker (Recomendado para Evaluación)
1. Ubicarse en la carpeta del proyecto y crear el archivo `.env` a partir de `.env.example` (si no existe):
   ```bash
   cp .env.example .env
   ```
2. Levantar toda la solución (base de datos MongoDB, contenedor de procesamiento ETL/ML/PDF/Jupyter y contenedor del dashboard en Streamlit) con **un solo comando**:
   ```bash
   docker compose up --build
   ```

### 5.2 Puertos Expuestos y Orden de Ejecución
El archivo `docker-compose.yml` coordina automáticamente el orden de arranque mediante `depends_on` y `healthcheck`:

| Orden | Servicio en Docker | Puerto Host $\rightarrow$ Contenedor | Rol y URL de Acceso |
|:---:|---|---|---|
| **1°** | `mongodb` (`mongo:7.0`) | `27017 -> 27017` | Base de datos NoSQL con volumen persistente `mongo_data` y `healthcheck` vía `mongosh`. |
| **2°** | `procesamiento` | `8888 -> 8888` | Espera a que `mongodb` esté `healthy`, ejecuta `python -m src.run_all` (descarga $\rightarrow$ ETL $\rightarrow$ Parquet $\rightarrow$ Modelo `.joblib` $\rightarrow$ MongoDB $\rightarrow$ Reporte PDF) y deja disponible **Jupyter Lab** en `http://localhost:8888`. |
| **3°** | `app` (Streamlit) | `8501 -> 8501` | Lee los archivos Parquet, el modelo `models/modelo_orinoco.joblib`, el documento de MongoDB y el PDF sin reentrenar. Disponible en **`http://localhost:8501`**. |

### 5.3 Cómo Verificar que Todo Funciona Correctamente
1. **Verificar estado de los 3 contenedores:**
   ```bash
   docker compose ps
   ```
   Los servicios `orinoco_mongodb`, `orinoco_procesamiento` y `orinoco_streamlit_app` deben aparecer en estado `Up` (`healthy` para MongoDB).
2. **Abrir el Dashboard Web en el navegador:**
   Ingresar a **`http://localhost:8501`** y recorrer las **5 pestañas**:
   - *1. Resumen (KPIs)*: verificar métricas del dataset (`271.196` registros crudos, `23.376` limpios) y del modelo ganador (`XGBoost Regressor`, MAE `0,4266 m`, mejora de `+53,01%` vs. Naive).
   - *2. Datos y Cubos OLAP*: filtrar la tabla limpia y los 2 cubos OLAP por año, estación, época hidrológica y fase ENSO.
   - *3. Visualizaciones Interactivas*: interactuar con la serie temporal y revisar las 6 gráficas de la Actividad 7.
   - *4. Modelo ML y Predicción en Vivo*: usar el formulario interactivo para predecir el nivel a +14 días sin reentrenar.
   - *5. Insights (MongoDB) y Reporte PDF*: revisar los 6 insights, las 4 recomendaciones, las 3 consultas a MongoDB y hacer clic en **"Descargar Reporte PDF Automático"**.
3. **Abrir el Notebook en Jupyter Lab:**
   Ingresar a **`http://localhost:8888`** y abrir `notebooks/Armas_Eduardo_BigData_ProyectoFinal.ipynb`.

### 5.4 Ejecución Directa en Entorno Local (sin Docker)
Si desea ejecutar o regenerar todos los artefactos localmente con Python / `uv`:
```bash
python -m src.run_all
streamlit run app/app.py
```

---

## 6. Diagnóstico Paso a Paso de un Problema de Infraestructura (Requisito 9)

Durante la integración de los tres contenedores en `docker-compose.yml` se presentó y resolvió el siguiente problema real de infraestructura:

- **Síntoma / Error observado:**
  Al ejecutar por primera vez `docker compose up --build`, el contenedor `procesamiento` y el contenedor `app` fallaban de forma intermitente en los primeros segundos con dos errores encadenados:
  1. `pymongo.errors.ServerSelectionTimeoutError: localhost:27017: [Errno 111] Connection refused` (o fallo al conectar a `mongodb:27017` antes de que el demonio `mongod` terminara de inicializar la autenticación del usuario `admin`).
  2. Condición de carrera (*race condition*) en el volumen compartido `./data/processed`: el contenedor `app` (Streamlit) arrancaba al mismo tiempo que `procesamiento` e intentaba leer `data/processed/orinoco_limpio.parquet` y `models/modelo_orinoco.joblib` antes de que el pipeline ETL/ML hubiera terminado de escribirlos.
- **Diagnóstico paso a paso:**
  1. Se inspeccionaron los registros de los contenedores con `docker compose logs mongodb procesamiento app`.
  2. Se identificó que, dentro de la red puente (`bridge`) de Docker (`bigdata_net`), `localhost` apunta al propio contenedor de Python y no al contenedor de MongoDB; el nombre de host DNS interno correcto es el nombre del servicio (`mongodb`).
  3. Además, `depends_on: [mongodb]` por defecto en Docker Compose solo espera a que el contenedor sea creado, **no** a que el motor `mongod` esté aceptando conexiones TCP autenticadas en el puerto `27017`.
- **Solución implementada:**
  1. Se parametrizó `MONGO_URI=mongodb://admin:bigdata2026@mongodb:27017/?authSource=admin` en `.env.example` / `.env` usando el nombre DNS del servicio `mongodb` dentro de la red `bigdata_net`.
  2. Se agregó un bloque `healthcheck` nativo en el servicio `mongodb` de `docker-compose.yml` que ejecuta `mongosh --eval "db.adminCommand('ping')"` cada 10 segundos, y se configuró `depends_on: mongodb: condition: service_healthy` tanto en `procesamiento` como en `app`.
  3. En el comando de arranque del servicio `app`, se añadió una guarda idempotente que verifica la existencia de `data/processed/orinoco_limpio.parquet` antes de levantar Streamlit, y en `src/mongodb_store.py` se implementó un mecanismo de respaldo espejo (`data/processed/mongodb_analysis_doc.json` + `mongomock`) para que el notebook y el dashboard funcionen sin errores incluso si se abren fuera de Docker.

---

## 7. Guía de Estructura para el Video de Sustentación (Mínimo 20 Minutos)

1. **Introducción (2–3 min):** Explicar el problema de pronóstico a 14 días en el río Orinoco, las 3 fuentes oficiales integradas (Niveles INAMEH/UCAB, 15 estaciones NASA POWER y NOAA ONI) y la justificación de la frecuencia diaria real (271.196 registros crudos sin datos sintéticos).
2. **Arquitectura (3–4 min):** Mostrar el diagrama de la Sección 3, explicar la decisión cuantitativa `pandas + PyArrow` vs. `Spark` (1,73 s de ETL y reducción de 60,0% de RAM) y describir los 3 servicios de `docker-compose.yml`.
3. **ETL y Visualizaciones (5–6 min):** Mostrar en el notebook los 6 problemas de calidad detectados (incluyendo los 31 días de sensores congelados y los 3 años bisiestos), los conteos antes/después de las 4 técnicas ETL, los 2 cubos OLAP y qué responde cada una de las 6 gráficas.
4. **Machine Learning (5–6 min):** Explicar cómo el embargo temporal de 14 días evita el *data leakage*, comparar las métricas en Test 2023–2024 (`Naive` MAE `0,9078 m` vs. `Ridge` `0,4505 m`, `Random Forest` `0,4347 m` y `XGBoost` `0,4266 m` con $R^2 = 0,9989$) e interpretar la importancia de variables en lenguaje de negocio.
5. **Demostración en Vivo (4–5 min):** Ejecutar `docker compose up --build`, recorrer las 5 pestañas del dashboard en `http://localhost:8501`, realizar una predicción nueva en el formulario interactivo y descargar el reporte PDF generado desde MongoDB.
6. **Aprendizajes (2 min):** Reflexionar sobre el impacto de detectar rachas planas ocultas, la ventaja de modelar incrementos $\Delta y_{t+14}$ entre estaciones de distinta altitud y el valor de empaquetar todo en Docker.
