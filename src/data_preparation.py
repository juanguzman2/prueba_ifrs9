# file: data_cleaner.py

import pandas as pd
import numpy as np

class DataCleaner:
    def __init__(self, lower_winsor=0.1, upper_winsor=0.9):
        self.lower_winsor = lower_winsor
        self.upper_winsor = upper_winsor

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._drop_irrelevant_columns(df)
        df = self._categorize_columns(df)
        df = self._winsorize_numeric(df)
        df = self._impute_missing_values(df)
        df = self._remove_duplicates(df)
        return df

    def _drop_irrelevant_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        cols_to_drop = ['num_doc', 'f_analisis','tipo_cliente']
        return df.drop(columns=cols_to_drop, errors='ignore')

    def _categorize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df['CO01MOR098RO'] = pd.cut(df['CO01MOR098RO'], bins=[-0.01, 25, 75, 100],
                                    labels=['bajo', 'medio', 'alto'])
        df['CO02MOR092TO'] = pd.cut(df['CO02MOR092TO'], bins=[-0.01, 25, 75, 100],
                                    labels=['bajo', 'medio', 'alto'])
        df['disp309'] = pd.cut(df['disp309'], bins=[0, 5, 9, 20],
                               labels=['bajo', 'medio', 'alto'])
        df['CO01NUM002AH'] = pd.cut(df['CO01NUM002AH'], bins=[-0.01, 0, 2, 5, 20],
                                    labels=['sin_ahorro', 'bajo', 'medio', 'alto'])
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
        categorical_with_nans = ['disp309', 'CO02MOR092TO', 'CO01MOR098RO']
        for col in categorical_with_nans:
            if pd.api.types.is_categorical_dtype(df[col]):
                if 'desconocido' not in df[col].cat.categories:
                    df[col] = df[col].cat.add_categories('desconocido')
            df[col] = df[col].fillna('desconocido')

        if df['CO01NUM002AH'].isnull().sum() > 0:
            df['CO01NUM002AH'] = df['CO01NUM002AH'].fillna(df['CO01NUM002AH'].mode()[0])
        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop_duplicates()
