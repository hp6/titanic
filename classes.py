import numpy as np
import pandas as pd
import joblib as jl

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.utils import Parallel, delayed



import functions as f



class DFNewFeaturesImputer(BaseEstimator, TransformerMixin):
    def __init__(self, inplace=False):
        self.inplace = inplace
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        if not self.inplace:
            X = X.copy()
        return X

class DFConflictingDataCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, inplace=False):
        self.inplace = inplace
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        if not self.inplace:
            X = X.copy()
        return X

class DFLogScaler(BaseEstimator, TransformerMixin):
    def __init__(self, columns=[]):
        self.columns = columns
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # scaled_df = pd.DataFrame(scaled_matrix, columns=self.columns)
        X.loc[:, self.columns] = np.log1p(X.loc[:, self.columns])
        return X

class DFScaler(BaseEstimator, TransformerMixin):
    def __init__(self, columns=[]):
        self.columns = columns
    
    def fit(self, X, y=None):
        self.scaler = StandardScaler().fit(X[self.columns])
        return self

    def transform(self, X, y=None):
        scaled_matrix = self.scaler.transform(X[self.columns])
        # scaled_df = pd.DataFrame(scaled_matrix, columns=self.columns)
        X[self.columns] = scaled_matrix
        return X



class DFOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=[], categories=[], sparse=False, handle_unknown="error"):
        self.columns = columns
        self.categories = categories
        self.sparse = sparse
        self.handle_unknown = handle_unknown
    
    def fit(self, X, y=None):
        if self.categories:
            self.one_hot_encoder = OneHotEncoder(categories=self.categories, sparse=self.sparse, handle_unknown=self.handle_unknown).fit(X[self.columns])
        else:
            self.categories = f.unique_values(X[self.columns])
            self.one_hot_encoder = OneHotEncoder(categories=self.categories, sparse=self.sparse, handle_unknown=self.handle_unknown).fit(X[self.columns])
        return self

    def transform(self, X, y=None):
        ohe_matrix = self.one_hot_encoder.transform(X[self.columns])
        column_names = []
        for index, col_categories in enumerate(self.categories):
            for cat in col_categories:
                column_names.append("{}_{}".format(self.columns[index], str(cat)))
        X.drop(self.columns, axis=1, inplace=True)
        return  pd.concat([X, pd.DataFrame(ohe_matrix, columns=column_names, index=X.index)], axis=1)

class DFNumToCat(BaseEstimator, TransformerMixin):
    def __init__(self, columns=[], inplace=False):
        self.columns = columns
        self.inplace = inplace
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if not self.inplace:
            X = X.copy()
        X[self.columns] = X[self.columns].astype(dtype="object")
        return X

class DFNaCatImputer(BaseEstimator, TransformerMixin):
    def __init__(self, fill_val="NA",  columns=[]):
        self.fill_val = fill_val
        self.columns = columns
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X[self.columns] = X[self.columns].fillna(self.fill_val)
        return X

class DFConstantImputer(BaseEstimator, TransformerMixin):
    def __init__(self, string_fill_val="NA", number_fill_val=0, columns=[], inplace=False):
        self.string_fill_val = string_fill_val
        self.number_fill_val = number_fill_val
        self.columns = columns
        self.inplace = inplace
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if not self.inplace:
            X = X.copy()
        str_col = X[self.columns].select_dtypes(include=object).columns
        num_col = X[self.columns].select_dtypes(include="number").columns
        # print(str_col)
        for row in str_col:
            X[row].fillna(self.string_fill_val, inplace=True)
        for row in num_col:
            X[row].fillna(self.number_fill_val, inplace=True)
        return X

class DFImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy="mean", columns=[], inplace=False):
        self.strategy = strategy
        self.columns = columns
        self.inplace = inplace
    
    def fit(self, X, y=None):
        self.imputer = SimpleImputer(strategy=self.strategy).fit(X.loc[:, self.columns])
        return self

    def transform(self, X, y=None):
        if not self.inplace:
            X = X.copy()

        X.loc[:, self.columns] = self.imputer.transform(X.loc[:, self.columns])
        return X

class DFSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns=[], inplace=False):
        self.columns = columns
        self.inplace = inplace
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        if not self.inplace:
            X = X.copy()

        if len(self.columns) == 1:
            return X[self.columns]
        else:
            return X[self.columns]
