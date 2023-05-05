import pandas as pd
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import RobustScaler;
from sklearn.impute import SimpleImputer;
from sklearn.pipeline import Pipeline


#Cargar conjunto de datos 
def load_datasets() -> pd.DataFrame:

    dataFrame = pd.read_csv("./train.csv");
    return dataFrame;
    

def main():

    dataFrame = load_datasets();
    dataFrame.drop(["Name"],axis="columns",inplace=True);

    from DataFramePreparerV2 import DataFramePreparer

    # DataFramePreparerV2
    DFPreparer = DataFramePreparer();
    DFPreparer.fit(dataFrame);
    dataFrame = DFPreparer.transform(dataFrame);
    print(dataFrame);

    # DataFramePreparerV1
    # DFPreparer = DataFramePreparer();
    # dataFrame = DFPreparer.transformData(dataFrame);

    best_accuracy = 0;
    best_f1_score = 0;
    print("Training model starting...");
    FILE_NAME = "logist_regression_titanic.sav";

    for i in range(1000):
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

        clf = LogisticRegression();
        clf.fit(X_train,Y_train);

        accuracy = clf.score(X_val,Y_val);
        if accuracy > best_accuracy:
            best_accuracy = accuracy;
            print("New best accuracy discovered",best_accuracy);

            #Making a predict to see if f1 score
            Y_pred = clf.predict(X_val);
            best_f1_score = f1_score(Y_val,Y_pred)
            print("F1 Score",best_f1_score);

            #Saving new model
            with open(FILE_NAME,"wb") as file:
                pickle.dump(clf,file);

    print("*******************");
    print("Training completed!");
    print("Best accuracy achieved:",best_accuracy);
    print("Best f1 score reached: ",best_f1_score);




if __name__ == "__main__":
    main();