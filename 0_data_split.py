# We are splitting training data into 80-20 for validation.
# 80% is now new training data.
# 20% is valdiation data

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


train = pd.read_csv("train.csv")
#target
y = train.hotel_cluster
train.drop(['hotel_cluster'], axis=1, inplace=True)
#splitting data
X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.2, random_state=42)
# adding train target to training data
X_train['hotel_cluster'] = y_train
# train_new.csv is 80% of original train.csv
X_train.to_csv('train_new.csv',index = False)
X_test.drop(['cnt', 'is_booking'], axis=1, inplace=True)
X_test.index = np.arange(0, len(X_test))
# test_new.csv is used as a validation data
X_test.to_csv('test_new.csv', index_label = 'id')
y_test = pd.DataFrame(y_test,columns=['hotel_cluster'])
y_test.index = np.arange(0, len(y_test))
# gnd_truth.csv is target labels for validation data
y_test.to_csv('gnd_truth.csv', index_label = 'id')

