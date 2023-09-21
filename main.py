import argparse

import pandas as pd
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


#Cargar conjunto de datos 
def load_datasets() -> pd.DataFrame:

    dataFrame = pd.read_csv("./train.csv");
    print("Dataset loaded!");
    return dataFrame;

#Train model
def train_model(dataFrame,model):
    best_accuracy = 0;
    best_f1_score = 0;
    print("Training model starting...");
    FILE_NAME = "model.sav";

    for _ in range(10):
        #Each time we are goin to create new test and train data
        train_set,validation_set = train_test_split(
            dataFrame,
            test_size=0.3,
            shuffle=True,
        );

        X_train = train_set.drop("Survived",axis=1);
        Y_train = train_set["Survived"].copy();

        X_val = validation_set.drop("Survived",axis=1); # -> pd.DataFrame
        Y_val = validation_set["Survived"].copy();

        model.fit(X_train,Y_train);

        #Making a predict to see if f1 score
        Y_pred = model.predict(X_val)

        accuracy_result = model.score(X_val,Y_val);
        f1_score_result = f1_score(Y_val,Y_pred);

        if f1_score_result > best_f1_score:
            best_accuracy = accuracy_result;
            best_f1_score = f1_score_result;

            print("New best accuracy achieved",best_accuracy);
            print("New best F1 Score achieved",best_f1_score);

                       

            #Saving new model
            with open(FILE_NAME,"wb") as file:
                pickle.dump(model,file);

    print("*******************");
    print("Training completed!");
    print("Best accuracy achieved:",best_accuracy);
    print("Best f1 score reached: ",best_f1_score);

    return best_f1_score,best_accuracy;

def main():

    #Defining all avaible algorithms to generate model
    ALGORITHMS = {
        "LogisticRegression":("Logistic Regression",LogisticRegression()),
        "SVMLinear":("SVM Linear",SVC(kernel="linear",C=50)),
        "SVMNoLinear":("SVM No linear",SVC(kernel="poly",degree=3,coef0=10,C=20)),
        "SVMGaussian":("Svm Gaussian",SVC(kernel="rbf",gamma=0.5,C=1000))
    };

    #Loading Dataset
    dataFrame = load_datasets();
    dataFrame.drop(["Name"],axis="columns",inplace=True);

    parser = argparse.ArgumentParser();
    parser.add_argument("--Algorithm",help = "Specific the algorithm that you wanna train");
    parser.add_argument("--DataFrameVersion",help = "Specific the version of Dataframe preparer you want to use");
    parser.add_argument("--All",help="Train with all the algoriths avaible: ");

    args = parser.parse_args();

    #DataFrame version to use
    if(args.DataFrameVersion == "V1"):

        print("Using DataFramePreparer V1...");
        from DataFramePreparerV1 import DataFramePreparer
        DFPreparer = DataFramePreparer();
        dataFrame = DFPreparer.transformData(dataFrame);

    if(args.DataFrameVersion == "V2"):

        print("Using DataFramePreparer V2...");
        from DataFramePreparerV2 import DataFramePreparer
        DFPreparer = DataFramePreparer();
        DFPreparer.fit(dataFrame);
        dataFrame = DFPreparer.transform(dataFrame);

    #Algorithm to use
    if(args.Algorithm == "All"):
        for nombre,(description,model) in ALGORITHMS.items():
            print("Using ",nombre," algorithm");
            best_f1_score,best_accuracy = train_model(dataFrame,model);

            #Register the results into the results.csv file
            results = pd.read_csv("results.csv");

            results.loc[results["Algorithm"] == nombre,["F1 Score","Accuracy"]] = [best_f1_score,best_accuracy];
            results.to_csv("results.csv");
        print("***************************************");
        print("Results");
        print("***************************************");
        print(results);
        exit();

    print("Using ",args.Algorithm," algorithm");
    model = ALGORITHMS[args.Algorithm][1];
    best_f1_score,best_accuracy = train_model(dataFrame,model);

    #Register the results into the results.csv file
    results = pd.read_csv("results.csv");
    # results.loc[results["Algorithm"] == args.Algorithm,"F1 Score"] = best_f1_score;
    # results.loc[results["Algorithm"] == args.Algorithm,"Accuracy"] = best_accuracy;

    results.loc[results["Algorithm"] == args.Algorithm,["F1 Score","Accuracy"]] = [best_f1_score,best_accuracy];
    print(results.loc[results["Algorithm"] == args.Algorithm]);
    results.to_csv("results.csv");

    '''
    .loc()
    dataframe.loc[etiqueta_fila, :]  
    # seleccionar todas las columnas para una fila determinada
    dataframe.loc[:, etiqueta_columna]  
    # seleccionar todas las filas para una columna determinada

    .iloc()
    dataframe.iloc[índice_fila, :]  # seleccionar todas las columnas para una fila determinada
    dataframe.iloc[:, índice_columna]  # seleccionar todas las filas para una columna determinada

    '''

    

if __name__ == "__main__":
    main();