from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer

import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

class DataFramePreparer(BaseEstimator,TransformerMixin):

    def __init__(self):

        self.numeric_attribs = None;
        self.category_attribs = None;
        self.full_pipeline = None;
        self.columns = None;

        self.numeric_pipeline = Pipeline([
            ("imputer",SimpleImputer(strategy="median")),
            ("rbst_scaler",RobustScaler())
        ]);

        self.category_pipeline = Pipeline([
            ("imputer",SimpleImputer(strategy="most_frequent")),
            ("oneHotEncoder",OneHotEncoder(sparse=False))
        ]);



    def fit(self,X:pd.DataFrame):

        self.columns = list(pd.get_dummies(X));

        self.numeric_attribs = X.select_dtypes(exclude = ["object"]);
        self.category_attribs = X.select_dtypes(include = ["object"]);

        
        self.full_pipeline = ColumnTransformer([
            ("num",self.numeric_pipeline,list(self.numeric_attribs.columns)),
            ("cat",self.category_pipeline,list(self.category_attribs.columns))
        ]);

        self.full_pipeline.fit(X);

        return self;
    
    def transform(self,X:pd.DataFrame):

        X_copy = X.copy();
        X_prep = self.full_pipeline.transform(X_copy);

        return pd.DataFrame(
            X_prep,
            columns = self.columns,
            # index = X.index
        );

        




