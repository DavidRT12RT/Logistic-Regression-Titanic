import pandas as pd;

from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler,OneHotEncoder

class CustomOneHotEnconder(BaseEstimator,TransformerMixin):

    def __init__(self):
        self.oh = OneHotEncoder();
        self.columns = None;

    def fit(self,X:pd.DataFrame):
        X_cat = X.select_dtypes(include=["object"]);
        self.columns = pd.get_dummies(X_cat);
        self.oh.fit(X_cat);
        return self;
    
    def transform(self,X:pd.DataFrame):
        X_copy = X.copy();
        X_cat = X_copy.select_dtypes(include=["object"]);
        X_cat_oh = self.oh.transform(X_cat);
        X_cat_oh = pd.DataFrame(
            X_cat_oh,
            columns = self.columns,
            index = X_copy.index
        );
        X_copy.drop(list(X_cat), axis=1, inplace=True)
        return X_copy.join(X_cat_oh)



class DataFramePreprarer():

    def __init__(self):

        self.numeric_attributes = None;
        self.category_attributes = None;
        self.columns = None;

        self.numeric_pipeline = Pipeline([
            #Valores nulos
            ("imputer",SimpleImputer(strategy="median")),
            #Estandarizacion
            ("rbst_scaler",RobustScaler())
        ]);

        self.category_pipeline = Pipeline([
            #Valores nulos
            ("imputer",SimpleImputer(strategy="most_frequent")),
            ("oneHotEnconder",CustomOneHotEnconder())
        ]);

        self.full_pipeline = None;


    def transformData(
        self,
        X:pd.DataFrame,
        category_columns = [],
        numeric_columns = []
    ) -> pd.DataFrame:


        self.numeric_attributes = list(X.select_dtypes(exclude=["object"]).columns);
        self.category_attributes = list(X.select_dtypes(include=["object"]).columns);

        if(len(category_columns) > 0):
            #User determine what category columns is going to apply transform
            self.category_attributes = category_columns;

        if(len(numeric_columns) > 0):
            #User determine what numeric columns is going to apply transform
            self.numeric_attributes = numeric_columns;


        self.full_pipeline = ColumnTransformer([
            ("num",self.numeric_pipeline,self.numeric_attributes),
            ("cat",self.category_pipeline,self.category_attributes)
        ]);

        X_prep = self.full_pipeline.fit_transform(X);
        self.columns = list(pd.get_dummies(X).columns);
        

        return pd.DataFrame(
            X_prep,
            columns = self.columns,
            index=X.index
        );












