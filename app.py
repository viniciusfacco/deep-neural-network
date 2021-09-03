from models import *
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

#read data
df = pd.read_csv('housepricedata.csv') #download from here: https://www.kaggle.com/moewie94/housepricedata?select=housepricedata.csv
dataset = df.values
#split features from labels
X = dataset[:,:10]
Y = dataset[:,10]
#normalize data
min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)
#split data in training and test
X_train, X_test, Y_train, Y_test = train_test_split(X_scale, Y, test_size=0.3)
X_train = X_train.T
X_test = X_test.T
Y_train = Y_train.reshape(Y_train.shape[0],-1).T
Y_test = Y_test.reshape(Y_test.shape[0],-1).T
#define the number of layers
layers_dims = [X.shape[1], 32, 32, 1]
#train
parameters, costs = L_layer_model(X_train, Y_train, layers_dims, learning_rate=0.25, num_iterations = 2500, print_cost = True)
#test
pred_test = L_model_predict_accuracy(X_test, Y_test, parameters)