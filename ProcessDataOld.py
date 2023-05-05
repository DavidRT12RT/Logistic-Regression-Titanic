from sklearn.base import BaseEstimator,TransformerMixin;
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
import pandas as pd

'''
En Scikit-learn, un Pipeline y un ColumnTransformer 
son dos clases que se utilizan para preprocesar los datos 
antes de ajustar un modelo de aprendizaje automático.

La principal diferencia entre un Pipeline y un 
ColumnTransformer es que un Pipeline se utiliza para 
encadenar varios pasos de preprocesamiento en una única 
tubería, mientras que un ColumnTransformer se utiliza 
para aplicar diferentes transformaciones a diferentes 
columnas en un solo paso.

Prepocesamiento de la data(
    Eliminacion de valores nulos(SimpleImputer),
    Regularacion (Normalizacion o estandarizacion)
    Convertir datos categoricos a numericos(One hot enconder)
)
'''

class DataFramePreparer(BaseEstimator,TransformerMixin):

    def __init__(self):
        self._full_pipeline = None;
        self._columns = None;
        self._numPipeline = Pipeline([
            #Valores nulos
            ("imputer",SimpleImputer(strategy="median")),
            #Escalar valores (estandarizacion con rango intercualitoco para eliminar outliners)
            ("rbst_scaler",RobustScaler())
        ]);
    
    def transformData(self,X_train:pd.DataFrame,Y=None):
        num_attribs = list(
            X_train.select_dtypes(exclude = ["object"])
        );
        cat_attribs = list(
            X_train.select_dtypes(include = ["object"])
        );
        self._full_pipeline = ColumnTransformer([
            ("num",self._numPipeline,num_attribs),
            ("cat",OneHotEncoder(),cat_attribs)
        ]);
        X_train_prep = self._full_pipeline.fit_transform(X_train);
        self._columns = pd.get_dummies(X_train);
        X_train_prep = pd.DataFrame(
            X_train_prep,
            columns=self._columns,
            # index=X_train.index
        );
        
        return X_train_prep;
