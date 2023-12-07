
import pandas as pd

breast_dataset = pd.read_csv("./TheBostonHousingDataset.csv")
for column in breast_dataset.columns:
    print(column,len(pd.unique(breast_dataset[column].values)))
# 'CHAS', 'RAD' are categorical features

def normalization(X):
    X = (X - X.mean())/X.std()
    return X
breast_dataset.drop(columns=['CHAS','RAD'],inplace=True,axis=1)
breast_dataset.apply(normalization)

one_hot_CHAS = pd.get_dummies(breast_dataset['CHAS'],dtype=float)
one_hot_CHAS.rename(columns={0:'CHAS_0',1:'CHAS_1'},inplace=True)
one_hot_RAD= pd.get_dummies(breast_dataset['RAD'],dtype=float)

X = pd.concat([breast_dataset,one_hot_CHAS,one_hot_RAD],axis=1)
print(X.columns)


from keras import Sequential
from keras.layers import Dense, Activation, Dropout, LeakyReLU
from keras.optimizers import Adam
from keras import metrics
from keras.losses import MeanSquaredError
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
# model is a 3-layer MLP with ReLU and dropout after each layer

hidden_units = 13
input_size = 22
dropout=0.1
num_labels=1

model = Sequential()
model.add(Dense(hidden_units, input_dim=input_size))
model.add(LeakyReLU())
model.add(Dropout(dropout))
model.add(Dense(hidden_units))
model.add(LeakyReLU())
model.add(Dropout(dropout))
model.add(Dense(num_labels))
model.add(Activation('relu'))
model.summary()
model.compile(loss=MeanSquaredError(),
                optimizer=Adam(learning_rate=0.0001),
                metrics=[metrics.mean_squared_error])


class boston_housing_model(Sequential):
    def __init__(self, hidden_units, input_size, dropout, num_labels):
        super().__init__()
        self.add(Dense(hidden_units, input_dim=input_size))
        self.add(LeakyReLU())
        self.add(Dropout(dropout))
        self.add(Dense(hidden_units,input_dim=hidden_units))
        self.add(LeakyReLU())
        self.add(Dropout(dropout))
        self.add(Dense(num_labels,input_dim=hidden_units))
        self.add(Activation('relu'))

'''
model_b = boston_housing_model(13,10,0.1,2)
model_b.summary()
model_b.fit()
'''

class BostonHousing:
    def __init__(self):
        # Load and preprocess the dataset
        breast_dataset = pd.read_csv("./TheBostonHousingDataset.csv")
        for column in breast_dataset.columns:
            print(column,len(pd.unique(breast_dataset[column].values)))
        # 'CHAS', 'RAD' are categorical features
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
        # Split the data
        train_val_x, test_x, train_val_y, test_y = train_test_split(X, y, test_size=0.1)
        train_x, val_x, train_y, val_y = train_test_split(train_val_x, train_val_y, test_size=0.2)

        self.train_data = (train_x.values, train_y.values)
        self.val_data = (val_x.values, val_y.values)
        self.test_data = (test_x.values, test_y.values)
        
    def train(self, model, epoch, batch_size):
        train_x, train_y = self.train_data
        model.fit(train_x, train_y, epochs=epoch, batch_size=batch_size, verbose=1)

    def validation(self, model):
        val_x, val_y = self.val_data
        val_loss = model.evaluate(val_x, val_y, verbose=0)
        print('\nValidation set: Average loss: {:.4f}'.format(val_loss[0]))

    def test(self, model):
        test_x, test_y = self.test_data
        predictions = model.predict(test_x)
        mse = mean_squared_error(test_y, predictions)
        mae = mean_absolute_error(test_y, predictions)

        print('\nTest set: Average MSE: {:.4f}, Average MAE: {:.4f}\n'.format(mse, mae))

    def main(self):
        lr = 1e-4
        epoch_num = 500
        batch_size = 32

        model_train = model
        model_train.compile(loss=MeanSquaredError(),
                optimizer=Adam(learning_rate=lr),
                metrics=[metrics.mean_squared_error])
        
        self.train(model=model_train, epoch=epoch_num, batch_size=batch_size)
        self.validation(model_train)
        self.test(model_train)
        model.save('model.h5')

BostonHousing().main()

