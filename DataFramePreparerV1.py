from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler,OneHotEncoder

import pandas as pd

class DataFramePreparer():

    def __init__(self):

        self.numeric_attribs = None;
        self.category_attribs = None;
        self.oh = OneHotEncoder();

        self.numeric_pipeline = Pipeline([
            ("imputer",SimpleImputer(strategy="median")),
            ("rbst_scaler",RobustScaler())
        ]);



    def transformData(self,X:pd.DataFrame) -> pd.DataFrame:

        self.numeric_attribs = X.select_dtypes(exclude = ["object"]);
        self.category_attribs = X.select_dtypes(include = ["object"]);

        #Numeric values preprocess
        numeric_attribs_preprocessed = pd.DataFrame(
            data = self.numeric_pipeline.fit_transform(self.numeric_attribs),
            columns = self.numeric_attribs.columns
        );

        #Category values preprocess
        imputerCategoric = SimpleImputer(strategy = "most_frequent");
        category_attribs_imputed = imputerCategoric.fit_transform(self.category_attribs);

        category_attribs_imputed = pd.DataFrame(
            data = category_attribs_imputed,
            columns = self.category_attribs.columns
        );

        codificacion = self.oh.fit_transform(
            category_attribs_imputed
        );

        columnas_dummies = pd.get_dummies(category_attribs_imputed).columns;

        category_attribs_preprocessed = pd.DataFrame(
            data = codificacion.toarray(),
            columns = columnas_dummies
        );

        return pd.concat(
            [numeric_attribs_preprocessed,category_attribs_preprocessed],
            axis = "columns"
        );

