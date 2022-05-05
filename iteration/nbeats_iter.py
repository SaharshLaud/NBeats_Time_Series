# -*- coding: utf-8 -*-
"""Dataset50_NBeats.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/12Exm2K_DDi3A4ySCVA0UoMZPiPkNmhW8

#**Nbeats Model Iteration for 50 time series data**
"""

import os
import numpy as np
import pandas as pd 
import seaborn as se 
import matplotlib.pyplot as plt 
import datetime
from nbeats_forecast import NBeats
from torch import optim
from sklearn.metrics import mean_absolute_error, mean_squared_error

for i in range(50):  
      na = str(i)    
      data = pd.read_csv("/content/drive/MyDrive/Machine Learning/dataset50/{n}.csv".format(n=na))
      data = data.drop(data.columns[[0]],axis = 1)
      df = data['timestamp']

      data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')


      # split a univariate dataset into train/test sets
      def train_test_split(data, n_test):
        return data[:n_test], data[n_test:]

      size = int(len(data)*0.75)
      train,val = train_test_split(data,size)


      model = NBeats(data = train.value.values.reshape((-1,1)), period_to_forecast=len(val),backcast_length=len(val),stack=[2,3],nb_blocks_per_stack=3,thetas_dims=[2,5])
      model.fit(epoch=10,optimiser=optim.AdamW(model.parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False))

      prediction = model.predict(predict_data=val.value.values.reshape((-1,1)))

      x = data.timestamp.values
      y = data.value.values

      plt.figure(figsize=(20, 5), dpi=80)
      plt.plot(x,y,label="original")
      plt.plot(x[len(train):],prediction,label="prediction")
      plt.xlabel("Timestamp")
      plt.ylabel("Value")
      plt.title('{name}.csv'.format(name = na))
      plt.legend(["original", "prediction"], loc ="lower right")



      my_path = os.path.abspath("/content/drive/MyDrive/Machine Learning/predictions_dataset50/") # Figures out the absolute path for you in case your working directory moves around.
      my_file = '{name}.png'.format(name = na)
      plt.savefig(os.path.join(my_path, my_file))

for i in range(50):  
      na = str(i)
      if(i==4):
        continue    
      data = pd.read_csv("/content/drive/MyDrive/Machine Learning/dataset50/{n}.csv".format(n=na))
      data = data.drop(data.columns[[0]],axis = 1)
      df = data['timestamp']

      data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')


      # split a univariate dataset into train/test sets
      def train_test_split(data, n_test):
        return data[:n_test], data[n_test:]

      size = int(len(data)*0.75)
      train,val = train_test_split(data,size)


      model = NBeats(data = train.value.values.reshape((-1,1)), period_to_forecast=len(val),backcast_length=len(val),stack=[2,3],nb_blocks_per_stack=3,thetas_dims=[2,5])
      model.fit(epoch=10,optimiser=optim.AdamW(model.parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False))

      prediction = model.predict(predict_data=val.value.values.reshape((-1,1)))
      
      test = val['value']
      print("{n}.csv".format(n=na))
      print("Mean Absolute Error {:.2f}".format(mean_absolute_error(test,prediction)))
      print("Mean Squared Error {:.2f}".format(mean_squared_error(test,prediction)))
      print("Root Mean Squared Error {:.2f}".format(np.sqrt(mean_squared_error(test,prediction))))

