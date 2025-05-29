from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import pandas as pd
import io
import os
import sys

# Definir rutas base
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(BASE_DIR, 'src')
DATA_PATH = os.path.join(BASE_DIR, 'data')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model.pkl')

# Agregar src al path para importar módulos personalizados
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from predict import RiskPredictor
from data_preprocesing import ModelPreprocessor

# Crear instancia de la API
app = FastAPI(
    title="API Predicción de Riesgo de Crédito",
    description="Clasifica clientes en grupos de riesgo t1 a t8 con base en modelo entrenado",
    version="1.0"
)

# Inicializar preprocesador y predictor
preprocessor = ModelPreprocessor(apply_cleaning=True, balanceo=False)
predictor = RiskPredictor(
    model_path=MODEL_PATH,
    preprocessor=preprocessor,
)

@app.post("/predict/")
async def predict_riesgo(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents), sep='|', encoding='utf-8')
    df_resultado = predictor.predict_from_dataframe(df)
    return df_resultado



