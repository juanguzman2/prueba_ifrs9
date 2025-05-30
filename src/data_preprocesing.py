# src/preprocessing/model_preprocessor.py

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from data_preparation import DataCleaner
import warnings
warnings.filterwarnings("ignore")

from config import FEATURES_PATH
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

selected_features = pd.read_csv(FEATURES_PATH)['feature'].tolist()

class ModelPreprocessor:
    def __init__(self, target_column='default', apply_cleaning=True, balanceo=True):
        self.balanceo = balanceo
        self.target_column = target_column
        self.apply_cleaning = apply_cleaning
        self.preprocessor = None
        self.feature_names_out = None

    def transform(self, df: pd.DataFrame, fit: bool = False) -> tuple:
        if self.apply_cleaning:
            cleaner = DataCleaner()
            df = cleaner.clean(df)

        # Separar X e y
        if self.target_column in df.columns:
            y = df[self.target_column]
            X = df.drop(columns=[self.target_column])
        else:
            y = None
            X = df.copy()

        # Separar por tipo
        num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

        # Reemplazo de inf por NaN en numéricas
        X[num_cols] = X[num_cols].replace([np.inf, -np.inf], np.nan)

        # Pipelines
        num_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        cat_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        if fit or self.preprocessor is None:
            self.preprocessor = ColumnTransformer([
                ('num', num_pipe, num_cols),
                ('cat', cat_pipe, cat_cols)
            ])
            X_processed = self.preprocessor.fit_transform(X)
            ohe_cols = self.preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(cat_cols)
            self.feature_names_out = num_cols + ohe_cols.tolist()
        else:
            X_processed = self.preprocessor.transform(X)

        # Crear DataFrame con nombres
        X_df = pd.DataFrame(X_processed, columns=self.feature_names_out)

        # Rellenar columnas faltantes con 0 y reordenar
        for col in self.feature_names_out:
            if col not in X_df.columns:
                X_df[col] = 0
        X_df = X_df[self.feature_names_out]

        # Aplicar balanceo solo si es entrenamiento
        if self.balanceo and y is not None and fit:
            under = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
            over = SMOTE(sampling_strategy='auto', random_state=42)
            imb_pipeline = ImbPipeline(steps=[
                ('under', under),
                ('over', over)
            ])
            X_resampled, y_resampled = imb_pipeline.fit_resample(X_df, y)
            return X_resampled, y_resampled

        return X_df, y
