# file: data_cleaner.py

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

class DataCleaner:
    def __init__(self, lower_winsor=0.1, upper_winsor=0.9):
        self.lower_winsor = lower_winsor
        self.upper_winsor = upper_winsor
        self.variables_log = [
            "trx39", "trx102", "trx106", "trx143", "trx158",
            "CO01END010RO", "CO01ACP017CC", "CO02EXP011TO", "CO02EXP004TO",
            "CO01EXP001CC", "CO01EXP003RO", "CO02END015CC", "CO01END002RO",
            "CO01END086RO", "CO01END094RO", "CO02NUM043RO", "CO01EXP002AH",
            "CO01END051RO", "CO01ACP011RO", "CO01MOR098RO", "CO02MOR092TO",
            "relacion_saldo_cupo", "variacion_saldo_9m", "utilizacion_actual",
            "delta_trx_mes"
        ]

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._generar_variables_derivadas(df)
        df = self._drop_irrelevant_columns(df)
        df = self._transformar_logaritmicamente(df)
        df = self._categorize_columns(df)
        df = self._winsorize_numeric(df)
        df = self._impute_missing_values(df)
        df = self._remove_high_null_columns(df)
        df = self._remove_duplicates(df)
        return df

    def _drop_irrelevant_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        cols_to_drop = ['num_doc', 'f_analisis', 'tipo_cliente']
        return df.drop(columns=cols_to_drop, errors='ignore')

    def _generar_variables_derivadas(self, df: pd.DataFrame) -> pd.DataFrame:
        df["f_analisis"] = df["f_analisis"].astype(str)
        df["anio"] = df["f_analisis"].str[:4].astype(int)
        df["mes"] = df["f_analisis"].str[4:].astype(int)
        df["antiguedad_dif_telcos"] = df["CO01EXP001CC"] - df["CO02EXP004TO"]
        df["antiguedad_dif_ahorro_rot"] = df["CO01EXP003RO"] - df["CO01EXP002AH"]
        df["utilizacion_actual"] = 1 - df["CO01END010RO"] / (df["CO01END094RO"] + 1e-6)
        df["variacion_saldo_9m"] = df["CO01END002RO"] - df["CO01END051RO"]
        df["relacion_saldo_cupo"] = df["CO01END002RO"] / (df["CO01END094RO"] + 1e-6)
        df["delta_trx_mes"] = df["trx102"] - df["trx106"]
        df["util_cat"] = pd.cut(df["utilizacion_actual"], bins=[-np.inf, 0.3, 0.7, np.inf], labels=["baja", "media", "alta"])
        df["mora_rot_cat"] = pd.cut(df["CO01MOR098RO"], bins=[-np.inf, 0.7, 0.9, 1.0], labels=["mala", "media", "buena"])
        df["mora_total_cat"] = pd.cut(df["CO02MOR092TO"], bins=[-np.inf, 0.7, 0.9, 1.0], labels=["mala", "media", "buena"])
        df["var_saldo_cat"] = pd.cut(df["variacion_saldo_9m"], bins=[-np.inf, 0, np.inf], labels=["deterioro", "mejora"])
        df["bandera_riesgo_1"] = ((df["variacion_saldo_9m"] < 0) & (df["utilizacion_actual"] > 0.7)).astype(int)
        df["bandera_oportunidad_telcos"] = ((df["CO01EXP001CC"] < 12) & (df["CO02MOR092TO"] > 0.9)).astype(int)
        return df

    def _transformar_logaritmicamente(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.variables_log:
            if col in df.columns:
                try:
                    df[f"log_{col}"] = np.log1p(df[col].astype(float))
                except Exception as e:
                    print(f"⚠️ No se pudo transformar '{col}': {e}")
        return df

    def _categorize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df['CO01MOR098RO'] = pd.cut(df['CO01MOR098RO'], bins=[-0.01, 25, 75, 100], labels=['bajo', 'medio', 'alto'])
        df['CO02MOR092TO'] = pd.cut(df['CO02MOR092TO'], bins=[-0.01, 25, 75, 100], labels=['bajo', 'medio', 'alto'])
        df['disp309'] = pd.cut(df['disp309'], bins=[0, 5, 9, 20], labels=['bajo', 'medio', 'alto'])
        df['CO01NUM002AH'] = pd.cut(df['CO01NUM002AH'], bins=[-0.01, 0, 2, 5, 20], labels=['sin_ahorro', 'bajo', 'medio', 'alto'])
        return df

    def _winsorize_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).drop(columns=['default'], errors='ignore').columns
        for col in numeric_cols:
            df[col] = self._winsorize_series(df[col])
        return df

    def _winsorize_series(self, s: pd.Series) -> pd.Series:
        lower_val = s.quantile(self.lower_winsor)
        upper_val = s.quantile(self.upper_winsor)
        return s.clip(lower=lower_val, upper=upper_val)

    def _impute_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        # Imputación de variables categóricas tipo binning
        for col in ['disp309', 'CO02MOR092TO', 'CO01MOR098RO', 'mora_rot_cat', 'mora_total_cat']:
            if col in df.columns and pd.api.types.is_categorical_dtype(df[col]):
                if 'desconocido' not in df[col].cat.categories:
                    df[col] = df[col].cat.add_categories('desconocido')
                df[col] = df[col].fillna('desconocido')

        if 'CO01NUM002AH' in df.columns and df['CO01NUM002AH'].isnull().sum() > 0:
            df['CO01NUM002AH'] = df['CO01NUM002AH'].fillna(df['CO01NUM002AH'].mode()[0])

        # Imputación de variables log-transformadas
        vars_alta_nulidad = [
            ("log_CO02END015CC", "CO02END015CC"),
            ("log_variacion_saldo_9m", "variacion_saldo_9m"),
            ("log_CO01ACP017CC", "CO01ACP017CC"),
            ("log_CO01ACP011RO", "CO01ACP011RO")
        ]
        vars_media_nulidad = [
            ("log_CO01END010RO", "CO01END010RO"),
            ("log_CO01END051RO", "CO01END051RO"),
            ("log_CO01END086RO", "CO01END086RO"),
            ("log_CO01END002RO", "CO01END002RO"),
            ("log_CO01END094RO", "CO01END094RO"),
            ("log_CO01EXP003RO", "CO01EXP003RO")
        ]
        vars_baja_nulidad = [
            ("log_CO02EXP011TO", "CO02EXP011TO"),
            ("log_CO02EXP004TO", "CO02EXP004TO"),
            ("log_relacion_saldo_cupo", "relacion_saldo_cupo"),
            ("log_delta_trx_mes", "delta_trx_mes")
        ]

        for log_col, base_col in vars_alta_nulidad + vars_media_nulidad:
            if log_col in df.columns:
                mediana = df[base_col].median()
                df[f"{log_col}_imputado"] = df[log_col].isna().astype(int)
                df[log_col] = df[log_col].fillna(np.log1p(mediana))

        for log_col, base_col in vars_baja_nulidad:
            if log_col in df.columns:
                df[log_col] = df[log_col].fillna(np.log1p(df[base_col].median()))
        
        return df

    def _remove_high_null_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        cols_a_eliminar = df.columns[df.isnull().mean() > 0.1]
        return df.drop(columns=cols_a_eliminar, errors='ignore')

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop_duplicates()
