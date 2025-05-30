

import pandas as pd
import numpy as np


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.utils import resample

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from statsmodels.stats.outliers_influence import variance_inflation_factor

from data_preparation import DataCleaner


from config import FEATURES_PATH
selected_features = pd.read_csv(FEATURES_PATH)['feature'].tolist()

class ModelPreprocessor:
    def __init__(self, target_column='default', apply_cleaning=True,balanceo=True):

        self.balanceo = balanceo
        self.target_column = target_column
        self.apply_cleaning = apply_cleaning
        self.preprocessor = None
        self.feature_names_out = None

    def transform(self, df: pd.DataFrame) -> tuple:
        if self.apply_cleaning:
            cleaner = DataCleaner()
            df = cleaner.clean(df)

        if self.balanceo:
            df_majority = df[df[self.target_column] == 0]
            df_minority = df[df[self.target_column] == 1]
            df_majority_downsampled = resample(
                df_majority,
                replace=False,
                n_samples=3 * len(df_minority),
                random_state=42
            )
            df = pd.concat([df_majority_downsampled, df_minority])
            df = df.sample(frac=1, random_state=42)

        if self.target_column in df.columns:
            y = df[self.target_column]
            X = df.drop(columns=[self.target_column])
        else:
            y = None
            X = df.copy()

        num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

        num_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        cat_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        self.preprocessor = ColumnTransformer([
            ('num', num_pipe, num_cols),
            ('cat', cat_pipe, cat_cols)
        ])

        X_processed = self.preprocessor.fit_transform(X)
        ohe_cols = self.preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(cat_cols)
        self.feature_names_out = num_cols + ohe_cols.tolist()
        X_df = pd.DataFrame(X_processed, columns=self.feature_names_out)

        X_df = X_df.loc[:, X_df.columns.intersection(selected_features)]
        return X_df, y


class FeatureSelector:
    def __init__(self, vif_threshold=10, corr_threshold=0.8):
        self.vif_threshold = vif_threshold
        self.corr_threshold = corr_threshold
        self.selected_features = []

    def remove_correlated(self, df: pd.DataFrame) -> pd.DataFrame:
        corr_matrix = df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > self.corr_threshold)]
        self.corr_pairs = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)).stack().reset_index()
        self.corr_pairs.columns = ['var1', 'var2', 'correlation']
        return df.drop(columns=to_drop), to_drop

    def remove_vif(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df.columns.tolist()
        while True:
            vif_data = pd.DataFrame()
            vif_data["feature"] = features
            vif_data["VIF"] = [variance_inflation_factor(df[features].values, i) for i in range(len(features))]
            max_vif = vif_data["VIF"].max()
            if max_vif > self.vif_threshold:
                drop_feat = vif_data.sort_values("VIF", ascending=False).iloc[0]["feature"]
                features.remove(drop_feat)
            else:
                break
        return df[features], list(set(df.columns) - set(features))

    def lasso_selection(self, df: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        model = LogisticRegression(penalty='l1', solver='liblinear', C=0.01)
        model.fit(df, y)
        selector = SelectFromModel(model, prefit=True)
        self.selected_features = df.columns[selector.get_support()].tolist()
        return df[self.selected_features]

    def select(self, df: pd.DataFrame, y: pd.Series, export_path: str = None) -> pd.DataFrame:
        df_corr, dropped_corr = self.remove_correlated(df)
        df_vif, dropped_vif = self.remove_vif(df_corr)
        df_lasso = self.lasso_selection(df_vif, y)

        if export_path:
            df_lasso.to_csv(export_path, index=False)

        print(f"Correlación alta eliminada: {len(dropped_corr)}")
        print(f"Colinealidad (VIF) eliminada: {len(dropped_vif)}")
        print(f"Variables finales (Lasso): {len(self.selected_features)}")
        return df_lasso
