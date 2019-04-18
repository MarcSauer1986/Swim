import pandas as pd
import os
import dask.dataframe as dd
import pickle
from virtual_swim_coach import *

# Make prediction

# Load data
user_input = input('Gebe den Ordnerpfad an, der alle Sensordaten (accel, gyro, magn) beinhaltet: ')
assert os.path.exists(user_input), "Ich konnte die Daten in folgendem Pfad nicht finden: " + str(user_input)
raw_accel = dd.read_csv(str(user_input) +'/accel*.csv', header=None).compute()
raw_gyro = dd.read_csv(str(user_input) +'/gyro*.csv', header=None).compute()
raw_magn = dd.read_csv(str(user_input) +'/magn*.csv', header=None).compute()
print('Sehr gut, es wurden alle Sensordaten gefunden!')

# Create data frame with all sensor information
column_names = ['Accel_X', 'Accel_Y', 'Accel_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z', 'Magn_X', 'Magn_Y', 'Magn_Z']
raw_data = pd.concat([raw_accel, raw_gyro, raw_magn], axis=1, sort=False)
raw_data.columns = column_names

# Create data frame for every single stroke and save as csv
clean_data = single_stroke(raw_data, user_input, save_data=True)

# ToDo: das ist nicht so schön, aber funktioniert. Noch anpassen.
condition_0 = 'irgendwas'
condition_1 = 'irgendwas'
no_condition = str(user_input) + '/stroke_*.csv'

# Feature engineering
clean_data = feature_engineering(no_condition)
pd.DataFrame(clean_data)

# Prepare data
X = clean_data

# Load the model from disk and make prediction
filename = 'model_logreg.sav'
model = pickle.load(open(filename, 'rb'))
y_pred_proba = model.predict_proba(X)
y_pred = model.predict(X)
print(y_pred_proba)
print(y_pred)
