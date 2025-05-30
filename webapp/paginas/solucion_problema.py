import streamlit as st


st.markdown("""
# 🧠 Prueba Analítica: Riesgo de Crédito IFRS9

## 📌 Introducción

Este proyecto tiene como objetivo el desarrollo de un modelo de predicción de incumplimiento de obligaciones financieras para clasificar clientes en ocho grupos de riesgo, definidos según rangos de probabilidad. El modelo se construyó a partir de información transaccional y crediticia de clientes con experiencia previa en el sistema financiero, pero con bajo uso de canales bancarios.

---

## 🔍 Objetivo del Proyecto

- Predecir la probabilidad de incumplimiento (`default`) para clientes del tipo `objetivo`.
- Clasificar cada cliente en uno de los grupos de riesgo [T1–T8] según su PD estimada.
- Maximizar el porcentaje de población **en rango** por grupo.
- Evaluar la capacidad de generalización con datos fuera de muestra.

---

## 🗂️ Repositorio del Proyecto

Para más detalles técnicos, código fuente y documentación, consulta el repositorio:

🔗 [https://github.com/juanguzman2/prueba_ifrs9](https://github.com/juanguzman2/prueba_ifrs9)

---

## 🧱 Ingeniería de Datos

### 🔧 Principales Procesos

- **Eliminación de columnas irrelevantes:** excluye identificadores y variables que inducen fuga de información.
- **Derivación de variables:** como `utilizacion_actual`, `relacion_saldo_cupo` y `delta_trx_mes`, útiles para capturar señales tempranas de riesgo.
- **Transformación logarítmica:** aplicada a variables sesgadas para estabilizar su distribución.
- **Discretización:** clasificación de variables en niveles (`bajo`, `medio`, `alto`) para facilitar reglas de negocio y segmentación.
- **Winsorización:** recorte del 10% superior e inferior en variables numéricas para mitigar outliers extremos.
- **Imputación de nulos:** mediante mediana, moda o categoría `desconocido`, con banderas auxiliares para trazabilidad.
- **Eliminación de duplicados y variables con alta nulidad.**

📄 Implementado en la clase `DataCleaner` del archivo [data_preparation.py](https://github.com/juanguzman2/prueba_ifrs9/blob/master/src/data_preparation.py)

---

## 🧪 Preprocesamiento para Modelado

La clase `ModelPreprocessor` gestiona todo el flujo de transformación previo al entrenamiento o predicción de modelos.

### Funcionalidades:

- **Limpieza opcional:** mediante `DataCleaner` (`apply_cleaning=True`).
- **Transformación de variables:**
  - Numéricas: imputación con mediana + escalado (`StandardScaler`)
  - Categóricas: imputación con moda + codificación (`OneHotEncoder`)
- **Manejo de columnas invisibles:** asegura consistencia entre entrenamiento y validación.
- **Control de `inf` y columnas faltantes.**

### ⚖️ Balanceo de Clases

Se detectó un desbalance severo en la clase `default`. Para solucionarlo:

- **Undersampling** con `RandomUnderSampler`
- **Oversampling** con `SMOTE`

✔️ Aplicado solo en entrenamiento (`fit=True`) para evitar sesgo en validación.

📄 Implementado en la clase `ModelPreprocessor` del archivo [model_preprocessor.py](https://github.com/juanguzman2/prueba_ifrs9/blob/master/src/data_preprocesing.py)

---

## 🤖 Selección y Entrenamiento del Modelo

Se aplicó un proceso robusto de experimentación, validación y selección del mejor modelo utilizando `MLflow`.

### 🔁 Flujo de trabajo

1. **Preprocesamiento:** usando `ModelPreprocessor` sobre `base_train.csv`
2. **Evaluación de modelos base:**
   - Logistic Regression
   - Random Forest
   - XGBoost
   - LightGBM
3. **Optimización:**
   - `GridSearchCV` con validación cruzada (3 folds)
   - Métrica objetivo: `F1-score`
4. **Registro en MLflow:**
   - Almacena hiperparámetros, métricas (`F1`, `Precision`, `Recall`, `AUC`) y artefactos serializados

### 📊 Comparación de Modelos: Preselección vs Optimización

| Modelo                 | F1 Score | Precisión | Recall |
|------------------------|----------|-----------|--------|
| LogisticRegression_HPO | 0.54     | 0.42      | 0.74   |
| RandomForest_HPO       | 0.54     | 0.47      | 0.62   |
| XGBoost_HPO            | **0.55** | **0.44**  | **0.75** |
| LightGBM_HPO           | 0.54     | 0.43      | 0.70   |
| **XGBoost_Optimized**  | 0.54     | 0.43      | 0.75   |

✅ **XGBoost Optimizado** fue seleccionado como modelo final por su desempeño balanceado y estabilidad con datos fuera de muestra (201901).

El modelo fue almacenado en un archivo `.pkl` para su uso posterior. La ubicacion del modelo es: [Model.pkl](https://github.com/juanguzman2/prueba_ifrs9/blob/master/models/model.pkl)

📈 Visualización de MLflow:
![MLflow Results](https://github.com/juanguzman2/prueba_ifrs9/blob/master/images/MLFlow.png?raw=true)

---

## 🔮 API de Predicción de Riesgo de Crédito

Una vez entrenado el modelo final, se implementó una API REST para consumo externo.

### 📥 Endpoint: `/predict/`

- **Método:** `POST`
- **Tipo de contenido:** `multipart/form-data`
- **Parámetro:** `file` — archivo `.csv` separado por `|` con la misma estructura de variables del entrenamiento.

📄 Lógica implementada en el archivo [app.py](https://github.com/juanguzman2/prueba_ifrs9/blob/master/api/app.py)

---

### 🧠 Lógica Interna de Predicción

La API utiliza la clase `RiskPredictor` para realizar todo el flujo de predicción, que incluye:

1. **Carga del modelo serializado** desde disco (`model.pkl`) usando `joblib`.
2. **Transformación de los datos** de entrada con una instancia del `ModelPreprocessor` (sin rebalanceo en producción).
3. **Predicción de probabilidad de incumplimiento** (`y_proba = model.predict_proba(X)[:, 1]`).
4. **Asignación de grupo de riesgo** (`t1` a `t8`) usando reglas basadas en rangos de PD definidos por negocio.
5. **Validación de que todas las predicciones caen en un grupo permitido**.

📄 Código fuente: [`predict.py`](https://github.com/juanguzman2/prueba_ifrs9/blob/master/src/predict.py)

---

### 📤 Respuesta esperada (formato JSON)

```json
[
  {
    "num_doc": "12345678",
    "probabilidad": 0.0324,
    "grupo_riesgo": "t4"
  },
  ...
]
```

---

## 📌 Recomendaciones y Datos Adicionales

Para mejorar aún más la efectividad del modelo, sería ideal incorporar:

- **Datos de scoring externos** (Buró de crédito)
- **Variables socioeconómicas** del cliente (ciudad, ingresos)
- **Datos temporales o secuenciales** (historial mensual de mora)
- **Alertas tempranas** en canales alternos (APPs, call center)
- **Variables macroeconómicas** (tasas de interés, inflación)

📉 Estos datos podrían obtenerse con bajo costo si ya forman parte del core transaccional o CRM de la entidad.

            
## 🧩 Estrategia Teórica para Disponibilizar el Modelo

Para facilitar el consumo del modelo por servicios externos o usuarios finales, se plantea la siguiente solución teórica:

- El modelo entrenado se **serializa en formato `.pkl`** para garantizar su reutilización y trazabilidad.
- Se expone una **API REST construida con FastAPI**, que permite recibir archivos de entrada (`.csv`) y retornar las predicciones de riesgo (`grupo_riesgo`).
- Para mejorar la interacción, se puede desarrollar una **interfaz web en FastAPI o Streamlit** que permita a usuarios cargar archivos, visualizar resultados y descargar reportes de clasificación.

Este enfoque asegura que los resultados del modelo sean fácilmente accesibles por sistemas web, móviles o procesos de negocio automatizados sin necesidad de redearrollar el modelo.

"""
            
           
            )