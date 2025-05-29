# src/prediction/predictor.py

import pandas as pd
import joblib

class RiskPredictor:
    def __init__(self, model_path, preprocessor, feature_file=None):
        self.model_path = model_path
        self.preprocessor = preprocessor  # instancia ya creada del preprocesador
        self.model = self._load_model()
        self.feature_file = feature_file

    def _load_model(self):
        return joblib.load(self.model_path)

    def _asignar_grupo(self, prob):
        if 0 <= prob <= 0.01:
            return 't1'
        elif 0.01 < prob <= 0.015:
            return 't2'
        elif 0.015 < prob <= 0.03:
            return 't3'
        elif 0.03 < prob <= 0.045:
            return 't4'
        elif 0.045 < prob <= 0.08:
            return 't5'
        elif 0.08 < prob <= 0.15:
            return 't6'
        elif 0.15 < prob <= 0.30:
            return 't7'
        elif 0.30 < prob <= 1.0:
            return 't8'
        else:
            return 'fuera_rango'

    def predict(self, input_path):
        # 1. Cargar datos
        df = pd.read_csv(input_path, sep='|', encoding='utf-8')

        # 2. Transformar datos
        X, _ = self.preprocessor.transform(df)

        # 3. Predicción
        y_proba = self.model.predict_proba(X)[:, 1]

        # 4. Construcción de resultados
        df_result = pd.DataFrame({
            'num_doc': df.loc[X.index, 'num_doc'].values,
            'probabilidad': y_proba
        })
        df_result['grupo_riesgo'] = df_result['probabilidad'].apply(self._asignar_grupo)

        # 5. Validación
        assert df_result['grupo_riesgo'].isin([
            't1', 't2', 't3', 't4', 't5', 't6', 't7', 't8'
        ]).all(), "❌ Hay categorías fuera de rango"

        return df_result

    def predict_from_dataframe(self, df):
        X, _ = self.preprocessor.transform(df)
        y_proba = self.model.predict_proba(X)[:, 1]

        df_result = pd.DataFrame({
            'num_doc': df.loc[X.index, 'num_doc'].values,
            'probabilidad': y_proba
        })
        df_result['grupo_riesgo'] = df_result['probabilidad'].apply(self._asignar_grupo)

        assert df_result['grupo_riesgo'].isin([
            't1','t2','t3','t4','t5','t6','t7','t8'
        ]).all(), "❌ Categorías fuera de rango"

        return df_result