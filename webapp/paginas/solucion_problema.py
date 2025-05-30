import streamlit as st


st.markdown("""
# 🧠 Prueba Analítica: Riesgo de Crédito IFRS9

## 📌 Introducción

Este proyecto tiene como objetivo el desarrollo de un modelo de predicción de incumplimiento de obligaciones financieras para clasificar clientes en ocho grupos de riesgo, definidos según rangos de probabilidad. El modelo se construye a partir de información transaccional y crediticia de clientes con experiencia previa en el sistema financiero, pero bajo uso de canales bancarios.

---

## 🔍 Objetivo del Proyecto

- Predecir la probabilidad de incumplimiento (`default`) para clientes `objetivo`.
- Clasificar cada cliente en uno de los grupos [T1-T8] según su PD estimada.
- Maximizar el porcentaje de población **en rango** por grupo.
- Evaluar generalización con datos fuera de muestra.

---

## Repositorio y Documentación
Para más detalles sobre el proyecto, puedes consultar el repositorio en [GitHub](https://github.com/juanguzman2/prueba_ifrs9)
            
## Ingenieria de Datos

### 🔧 Principales Procesos

- **Eliminación de columnas irrelevantes:** remueve identificadores y variables que inducen fuga.
- **Derivación de variables:** crea nuevas features como `utilizacion_actual`, `relacion_saldo_cupo` o `delta_trx_mes`, útiles para medir señales de riesgo.
- **Transformación logarítmica:** normaliza distribuciones sesgadas y reduce impacto de valores extremos.
- **Discretización de variables:** clasifica en categorías como `bajo`, `medio`, `alto` para facilitar reglas de negocio.
- **Winsorización:** recorte del 10% inferior y superior en variables numéricas para mitigar outliers.
- **Imputación de valores nulos:** usa mediana, moda o categorías `desconocido`, y añade banderas de imputación.
- **Eliminación de duplicados y columnas con alta nulidad.**

Todo esta almacenado en la clase Datacleaner en el archivo [data_preparation.py](https://github.com/juanguzman2/prueba_ifrs9/blob/master/src/data_preparation.py)       

## 🧪 Preprocesamiento para Modelado

La clase `ModelPreprocessor` gestiona el flujo completo de transformación de variables antes de entrenar o predecir con modelos de riesgo de crédito.

### 🔧 Funcionalidades Principales

- **Limpieza opcional:** aplica el proceso de `DataCleaner` si se activa el parámetro `apply_cleaning=True`.
- **Separación X / y:** identifica la variable objetivo (`default`) y separa las variables predictoras.
- **Transformación numérica:**
  - Imputación de valores nulos con mediana.
  - Escalado con `StandardScaler` para normalizar magnitudes.
- **Transformación categórica:**
  - Imputación con la moda.
  - Codificación One-Hot (`OneHotEncoder`).

### ⚖️ Balanceo de Clases

Se encontro un gran desbalanceo de datos en la clase objetivo `Defaul`. Por lo tanto se opto por aplicar un balanceo de clases durante el entrenamiento del modelo.

Durante el entrenamiento, si `balanceo=True`, aplica una combinación:
- **Undersampling** con `RandomUnderSampler`.
- **Oversampling** con `SMOTE`.

Esto ayuda a mejorar el aprendizaje sobre la clase minoritaria (`default = 1`).

Todo esta almacenado en la clase ModelPreprocessor en el archivo [model_preprocessor.py](https://github.com/juanguzman2/prueba_ifrs9/blob/master/src/data_preprocesing.py)

## 🤖 Selección y Entrenamiento del Modelo
            

Para seleccionar el mejor modelo predictivo se aplicó un enfoque sistemático con validación, optimización de hiperparámetros y registro de resultados usando **MLflow**.

### ⚙️ Flujo de trabajo

1. **Carga y transformación de datos**:
   - Se aplicó el `ModelPreprocessor` para limpiar, transformar y balancear la data (`base_train.csv`).
   - Los datos fueron divididos en `train` y `test` con `stratify` para mantener la proporción de la clase objetivo (`default`).

2. **Modelos evaluados**:
   - `Logistic Regression`
   - `Random Forest`
   - `XGBoost`
   - `LightGBM`

3. **Optimización de hiperparámetros**:
   - Se utilizó `GridSearchCV` con validación cruzada de 3 folds y métrica objetivo `F1-score`.
   - Para el modelo final (`Random Forest`), se aplicó una grilla extendida con combinaciones avanzadas de parámetros como `max_depth`, `min_samples_split`, `max_features`, etc.

4. **Registro con MLflow**:
   - Cada experimento se registró con nombre, parámetros, métricas (`F1`, `Precision`, `Recall`, `AUC`) y el modelo serializado para reutilización.

### 📈 Mejores Resultados en Validación

Modelo final entrenado: **Random Forest Optimizado**

Métricas en muestra de validación (`base_validacion.csv`):

- **F1-score**: `0.53`
- **Precision**: `0.47`
- **Recall**: `0.60`
- **AUC**: `0.76`

### 🧪 Validación Final

Se aplicó el modelo a la muestra de enero 2019 usando el preprocesador ya entrenado (`fit=False`) para simular condiciones de producción. Esto garantiza consistencia y generalización del pipeline.

Este proceso asegura la selección del mejor modelo bajo restricciones de balanceo, interpretabilidad y desempeño esperadas en el contexto de riesgo de crédito.




            
"""
            )