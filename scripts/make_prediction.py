import pandas as pd
import os
import dask.dataframe as dd
import pickle
from virtual_swim_coach import *
import sys
import subprocess

# Make prediction

# Load data
user_input_1 = input('Gebe den Ordnerpfad an, der alle Sensordaten (accel, gyro, magn) beinhaltet: ')
assert os.path.exists(user_input_1), "Ich konnte die Daten in folgendem Pfad nicht finden: " + str(user_input_1)
raw_accel = dd.read_csv(str(user_input_1) +'/accel*.csv', header=None).compute()
raw_gyro = dd.read_csv(str(user_input_1) +'/gyro*.csv', header=None).compute()
raw_magn = dd.read_csv(str(user_input_1) +'/magn*.csv', header=None).compute()
print('Sehr gut, es wurden alle Sensordaten gefunden!')

# Create data frame with all sensor information
column_names = ['Accel_X', 'Accel_Y', 'Accel_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z', 'Magn_X', 'Magn_Y', 'Magn_Z']
raw_data = pd.concat([raw_accel, raw_gyro, raw_magn], axis=1, sort=False)
raw_data.columns = column_names

# Create data frame for every single stroke and save as csv
single_stroke(raw_data, user_input, save_data=True, print_status=False)

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

# Another prediction
user_input_2 = input("Möchtest du eine weitere Vorhersage machen?\n0: Nein\n1: Ja")
if user_input_2 == '0':
    subprocess.call(['python', 'virtual_swim_coach.py'])
elif user_input_2 == '1':
    os.execl(sys.executable, sys.executable, *sys.argv)
