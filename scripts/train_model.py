import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pickle
import subprocess
import dask.dataframe as dd
from virtual_swim_coach import *
import os
import sys

# Train model

# Prepare date
# Load data
user_input_1 = input("Gebe den Ordnerpfad an, der die Sensordaten (accel, gyro, magn) der Bedingung 'deep catch' beinhaltet: ")
assert os.path.exists(user_input_1), "Ich konnte die Daten in folgendem Pfad nicht finden: " + str(user_input_1)
raw_accel_0 = dd.read_csv(str(user_input_1) +'/accel*.csv', header=None).compute()
raw_gyro_0 = dd.read_csv(str(user_input_1) +'/gyro*.csv', header=None).compute()
raw_magn_0 = dd.read_csv(str(user_input_1) +'/magn*.csv', header=None).compute()
print("Sehr gut, es wurden alle Sensordaten der Bedingung 'deep catch' gefunden!")

user_input_2 = input("Gebe den Ordnerpfad an, der die Sensordaten (accel, gyro, magn) der Bedingung 'high elbow catch' beinhaltet: ")
assert os.path.exists(user_input_2), "Ich konnte die Daten in folgendem Pfad nicht finden: " + str(user_input_2)
raw_accel_1 = dd.read_csv(str(user_input_2) + '/accel*.csv', header=None).compute()
raw_gyro_1 = dd.read_csv(str(user_input_2) + '/gyro*.csv', header=None).compute()
raw_magn_1 = dd.read_csv(str(user_input_2) + '/magn*.csv', header=None).compute()
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
    single_stroke(raw_data_0, user_input_1, save_data=False, print_status=True)
    single_stroke(raw_data_1, user_input_2, save_data=False, print_status=True)
elif user_input_3 == '1':
    single_stroke(raw_data_0, user_input_1, save_data=True, print_status=True)
    single_stroke(raw_data_1, user_input_2, save_data=True, print_status=True)

# Define which data is relevant for training
user_input_4 = input('Soll das Modell die gesamte Datenhistorie berücksichtigen?\n0 Nein\n1 Ja\n')
if user_input_4 == '0':
    condition_0 = user_input_1 + '/stroke_*.csv'
    condition_1 = user_input_2 + '/stroke_*.csv'
if user_input_4 == '1':
    condition_0 = '/users/marcsauer/PycharmProjects/Swim/data/Run_0_*/stroke_*.csv'
    condition_1 = '/users/marcsauer/PycharmProjects/Swim/data/Run_1_*/stroke_*.csv'

# Feature engineering
feature_df_0 = feature_engineering(condition_0)
feature_df_0['condition'] = 0

feature_df_1 = feature_engineering(condition_1)
feature_df_1['condition'] = 1

frames = [feature_df_0, feature_df_1]
clean_data = pd.concat(frames)

# Save clean_data to CSV
pd.DataFrame(clean_data).to_csv('/users/marcsauer/PycharmProjects/Swim/data/clean_data.csv', index=False)
print("'clean_data' als CSV abgespeichert.")
print('Datenvorbereitung abgeschlossen!')

# Prepare data
X = np.array(clean_data['Accel_X_mean']).reshape(-1, 1)
# X = clean_data.drop(['condition'], axis=1)
y = clean_data['condition']

# Split data in train, validate and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
X_train = pd.DataFrame(X_train)
X_val = pd.DataFrame(X_val)

# Run training
# logreg = LogisticRegression(solver='liblinear', multi_class='ovr')
# model_logreg = logreg.fit(X_train, y_train)
lda = LinearDiscriminantAnalysis()
model_lda = lda.fit(X_train, y_train)
print('Training abgeschlossen!')

# Evaluate training
# score_logreg = model_logreg.score(X_val, y_val)
score_lda = model_lda.score(X_val, y_val)
print('Die Genauigkeit des Modells liegt bei: {:.2f}%'.format(score_lda * 100))

# Save model
user_input_5 = input('Soll das Modell gespeichert werden?\n0: Nein\n1: Ja')
if user_input_5 == '0':
    user_input_6 = input('Soll ein weiteres Modell trainiert werden?\n0: Nein\n1: Ja')
    if user_input_6 == '0':
        subprocess.call(['python', 'virtual_swim_coach.py'])
    elif user_input_6 == '1':
        os.execl(sys.executable, sys.executable, *sys.argv)
elif user_input_5 == '1':
    # filename = 'model_logreg.sav'
    filename = 'model_lda.sav'
    pickle.dump(model_lda, open(filename, 'wb'))
    print('Speichern des Modells abgeschlossen!')

# Make prediction
user_input_7 = input("Soll eine Vorhersage gemacht werden?\n0: Nein\n1: Ja")
if user_input_7 == '0':
    subprocess.call(['python', 'virtual_swim_coach.py'])
elif user_input_7 == '1':
    subprocess.call(['python', 'make_prediction.py'])
