import pandas as pd
import os
import dask.dataframe as dd
import subprocess
from virtual_swim_coach import *

# Prepare date

# Load data
user_input_1 = input("Geben Sie den Ordnerpfad an, der alle Sensordaten (accel, gyro, magn) der Bedingung 'deep catch' beinhaltet: ")
assert os.path.exists(user_input_1), "Ich konnte die Daten in folgendem Pfad nicht finden: " + str(user_input_1)
raw_accel_0 = dd.read_csv(str(user_input_1) +'/accel-*.csv', header=None).compute()
raw_gyro_0 = dd.read_csv(str(user_input_1) +'/gyro-*.csv', header=None).compute()
raw_magn_0 = dd.read_csv(str(user_input_1) +'/magn-*.csv', header=None).compute()
print("Sehr gut, es wurden alle Sensordaten der Bedingung 'deep catch' gefunden!")

user_input_2 = input("Geben Sie den Ordnerpfad an, der alle Sensordaten (accel, gyro, magn) der Bedingung 'high elbow catch' beinhaltet: ")
assert os.path.exists(user_input_2), "Ich konnte die Daten in folgendem Pfad nicht finden: " + str(user_input_2)
raw_accel_1 = dd.read_csv(str(user_input_2) + '/accel-*.csv', header=None).compute()
raw_gyro_1 = dd.read_csv(str(user_input_2) + '/gyro-*.csv', header=None).compute()
raw_magn_1 = dd.read_csv(str(user_input_2) + '/magn-*.csv', header=None).compute()
print('Hooray, wir haben alle deine Sensordaten gefunden!')

# Create data frame with all sensor information
column_names = ['Accel_X', 'Accel_Y', 'Accel_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z', 'Magn_X', 'Magn_Y', 'Magn_Z']
raw_data_0 = pd.concat([raw_accel_0, raw_gyro_0, raw_magn_0], axis=1, sort=False)
raw_data_0.columns = column_names

raw_data_1 = pd.concat([raw_accel_1, raw_gyro_1, raw_magn_1], axis=1, sort=False)
raw_data_1.columns = column_names

# Create data frame for every single stroke and save as csv
user_input_3 = input('Sollen die Schwimmzüge einzeln abgespeichert werden?\n0 Nein\n1 Ja\n')
if user_input_3 == '0':
    single_stroke(raw_data_0, user_input_1, save_data=False)
    single_stroke(raw_data_1, user_input_2, save_data=False)
elif user_input_3 == '1':
    single_stroke(raw_data_0, user_input_1, save_data=True)
    single_stroke(raw_data_1, user_input_2, save_data=True)
else:
    print('Sie müssen entweder 0 (Nein) oder 1 (Ja) eingeben.')

condition_0 = '/users/marcsauer/PycharmProjects/Swim/data/Run_0_*/stroke_*.csv'
condition_1 = '/users/marcsauer/PycharmProjects/Swim/data/Run_1_*/stroke_*.csv'

# condition_0 = str(user_input_1)+'/stroke_*.csv'
# condition_1 = str(user_input_2)+'/stroke_*.csv'

# Feature engineering
feature_df_0 = feature_dataframe(condition_0)
feature_df_1 = feature_dataframe(condition_1)
frames = [feature_df_0, feature_df_1]
feature_df_clean = pd.concat(frames)

# Save clean_data to CSV
pd.DataFrame(feature_df_clean).to_csv('/Users/marcsauer/PycharmProjects/Swim/data/clean_data.csv', index=False)
print("'clean_data' als CSV abgespeichert.")
print('Datenvorbereitung abgeschlossen!')
print("Soll das Modell nun trainiert werden?\n0: Nein\n1: Ja")
user_input_4 = input()
if user_input_4 == '1':
    print('Das Modell wird nun trainiert!')
    subprocess.call(['python', 'train_model.py'])
if user_input_4 == '0':
    pass
