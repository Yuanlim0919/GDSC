from tensorflow import keras
import pandas as pd
from sklearn.metrics import mean_squared_error

def data_preprocess():
    breast_dataset = pd.read_csv("./TheBostonHousingDataset.csv")

    def normalization(X):
        X = (X - X.mean())/X.std()
        return X
    one_hot_CHAS = pd.get_dummies(breast_dataset['CHAS'],dtype=float)
    one_hot_CHAS.rename(columns={0:'CHAS_0',1:'CHAS_1'},inplace=True)
    one_hot_RAD= pd.get_dummies(breast_dataset['RAD'],dtype=float)
    y = breast_dataset['MEDV']

    breast_dataset.drop(columns=['CHAS','RAD','MEDV'],inplace=True,axis=1)
    breast_dataset.apply(normalization)

    X = pd.concat([breast_dataset,one_hot_CHAS,one_hot_RAD],axis=1)
    return X,y

def load_model(model_path):
    model = keras.models.load_model(model_path)
    model.summary()
    return model

def main():
    X,y = data_preprocess()
    model_pred = keras.models.load_model('model.h5')
    y_pred = model_pred.predict(X)
    mseloss = mean_squared_error(y,y_pred)
    print(mseloss)

main()

