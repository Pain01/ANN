

#Artificial neural network
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Churn_Modelling.csv'
                     )

X = dataset.iloc[:,3:13].values
y = dataset.iloc[:,13].values
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

l1 = LabelEncoder()
l2 = LabelEncoder()
X [:,1] = l1.fit_transform(X[:,1])
X [:,2] = l2.fit_transform(X[:,2])
O = OneHotEncoder(categorical_features=[1])
X = O.fit_transform(X).toarray()
X = X[:,1:]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state = 0)


from sklearn.preprocessing import StandardScaler
sc =StandardScaler()
X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)

#ANN
#import lib
import keras
from keras.models import Sequential
from keras.layers import Dense

#Initializing ANN
Cr = Sequential()
#Addigng input layers and first hiidden layers
Cr.add(Dense(output_dim = 6,init = 'uniform',activation = 'relu',input_dim = 11))

Cr.add(Dense(output_dim = 6,init = 'uniform',activation = 'relu'))

#op  layer
Cr.add(Dense(output_dim = 1,init = 'uniform',activation = 'sigmoid'))

#Compiling ANN
Cr.compile(optimizer='adam',loss ='binary_crossentropy',metrics = ['accuracy'])
#Fit
Cr.fit(X_train,y_train,batch_size=10 ,nb_epoch = 100)

y_pred = Cr.predict(X_test)

y_pred = (y_pred>0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,)













