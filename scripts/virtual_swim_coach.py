import pandas as pd
import numpy as np
import glob
from scipy.signal import find_peaks
import os
import dask.dataframe as dd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import pickle

# Create data frame for every single stroke and save as csv
def single_stroke(raw_data, user_input_2, save_data=False):

    # Find best distance for stroke_cut
    var = []
    for i in range(1, 50):
        stroke_cut_iter, _ = (find_peaks(raw_data['Magn_Z'] * (-1), distance=i))
        # Lenght of single strokes
        stroke_length_iter = np.diff(stroke_cut_iter)
        variance = np.var(stroke_length_iter)
        var.append(variance)
    best_distance = var.index(min(var))

    # Cut stroke
    stroke_cut, _ = (find_peaks(raw_data['Magn_Z'] * (-1), distance=best_distance))
    stroke_length = np.diff(stroke_cut).tolist()
    if np.var(stroke_length) < 10:
        print('Schwimmzugerkennung: sehr gut (geringe Varianz): {:.2f}'.format(np.var(stroke_length)))
    else:
        print('Schwimmzugerkennung: okay. Varianz: ' + str(np.var(stroke_length)))
        print('Geringste Anzahl von Datenpunkte pro Zug: ' + str(min(stroke_length)))
        print('Höchste Anzahl von Datenpunkte pro Zug: ' + str(max(stroke_length)))

    # Drop outliers and create list of data frames
    criterion_outlier_lower_limit = (np.mean(stroke_length) - 2 * np.std(stroke_length))
    criterion_outlier_upper_limit = (np.mean(stroke_length) + 2 * np.std(stroke_length))
    stroke_cut_mod = [-1] + stroke_cut
    list_of_dfs = [raw_data.iloc[stroke_cut_mod[n] + 1:stroke_cut_mod[n + 1] + 1] for n in
                   range(len(stroke_cut_mod) - 1)]
    list_of_dfs_without_outliers = []
    for idx in range(0, len(stroke_length)):
        if stroke_length[idx] > criterion_outlier_lower_limit and stroke_length[idx] < criterion_outlier_upper_limit:
            list_of_dfs_without_outliers.append(list_of_dfs[idx])

    # Save single strokes to csv
    if save_data == True:
        k = 1
        for i in range(0, len(list_of_dfs_without_outliers)):
            pd.DataFrame(list_of_dfs_without_outliers[i]).to_csv(
                '{}/stroke_{}.csv'.format(user_input_2, k), index=False)
            k += 1
        print('Einzelne Züge als CSV abgespeichert.')

# Feature engineering
def feature_dataframe(condition):
    '''Creating features from single stroke and merge into new data frame'''
    Magn_X_mean = []
    Magn_X_min = []
    Magn_X_max = []
    Magn_Y_mean = []
    Magn_Y_min = []
    Magn_Y_max = []
    Magn_Z_mean = []
    Magn_Z_min = []
    Magn_Z_max = []
    Gyro_X_max = []
    Gyro_X_mean = []
    Gyro_X_min = []
    Gyro_Y_mean = []
    Gyro_Y_min = []
    Gyro_Y_max = []
    Gyro_Z_mean = []
    Gyro_Z_min = []
    Gyro_Z_max = []
    Accel_X_max = []
    Accel_X_mean = []
    Accel_X_min = []
    Accel_Y_mean = []
    Accel_Y_min = []
    Accel_Y_max = []
    Accel_Z_mean = []
    Accel_Z_min = []
    Accel_Z_max = []

    # Load data
    for file in glob.glob(condition):
        single_stroke = pd.read_csv(file)

        # Feature engineering
        # Magn_X_mean
        single_stroke_magn_x_mean = single_stroke["Magn_X"].mean()
        Magn_X_mean.append(single_stroke_magn_x_mean)

        # Magn_X_min
        single_stroke_magn_x_min = single_stroke["Magn_X"].min()
        Magn_X_min.append(single_stroke_magn_x_min)

        # Magn_X_max
        single_stroke_magn_x_max = single_stroke["Magn_X"].max()
        Magn_X_max.append(single_stroke_magn_x_max)

        # Magn_Y_mean
        single_stroke_magn_y_mean = single_stroke["Magn_Y"].mean()
        Magn_Y_mean.append(single_stroke_magn_y_mean)

        # Magn_Y_min
        single_stroke_magn_y_min = single_stroke["Magn_Y"].min()
        Magn_Y_min.append(single_stroke_magn_y_min)

        # Magn_Y_max
        single_stroke_magn_y_max = single_stroke["Magn_Y"].max()
        Magn_Y_max.append(single_stroke_magn_y_max)

        # Magn_Z_mean
        single_stroke_magn_z_mean = single_stroke["Magn_Z"].mean()
        Magn_Z_mean.append(single_stroke_magn_z_mean)

        # Magn_Z_min
        single_stroke_magn_z_min = single_stroke["Magn_Z"].min()
        Magn_Z_min.append(single_stroke_magn_z_min)

        # Magn_Z_max
        single_stroke_magn_z_max = single_stroke["Magn_Z"].max()
        Magn_Z_max.append(single_stroke_magn_z_max)

        # Gyro_X_mean
        single_stroke_gyro_x_mean = single_stroke["Gyro_X"].mean()
        Gyro_X_mean.append(single_stroke_gyro_x_mean)

        # Gyro_X_min
        single_stroke_gyro_x_min = single_stroke["Gyro_X"].min()
        Gyro_X_min.append(single_stroke_gyro_x_min)

        # Gyro_X_max
        single_stroke_gyro_x_max = single_stroke["Gyro_X"].max()
        Gyro_X_max.append(single_stroke_gyro_x_max)

        # Gyro_Y_mean
        single_stroke_gyro_y_mean = single_stroke["Gyro_Y"].mean()
        Gyro_Y_mean.append(single_stroke_gyro_y_mean)

        # Gyro_Y_min
        single_stroke_gyro_y_min = single_stroke["Gyro_Y"].min()
        Gyro_Y_min.append(single_stroke_gyro_y_min)

        # Gyro_Y_max
        single_stroke_gyro_y_max = single_stroke["Gyro_Y"].max()
        Gyro_Y_max.append(single_stroke_gyro_y_max)

        # Gyro_Z_mean
        single_stroke_gyro_z_mean = single_stroke["Gyro_Z"].mean()
        Gyro_Z_mean.append(single_stroke_gyro_z_mean)

        # Gyro_Z_min
        single_stroke_gyro_z_min = single_stroke["Gyro_Z"].min()
        Gyro_Z_min.append(single_stroke_gyro_z_min)

        # Gyro_Z_max
        single_stroke_gyro_z_max = single_stroke["Gyro_Z"].max()
        Gyro_Z_max.append(single_stroke_gyro_z_max)

        # Accel_X_mean
        single_stroke_accel_x_mean = single_stroke["Accel_X"].mean()
        Accel_X_mean.append(single_stroke_accel_x_mean)

        # Accel_X_min
        single_stroke_accel_x_min = single_stroke["Accel_X"].min()
        Accel_X_min.append(single_stroke_accel_x_min)

        # Accel_X_max
        single_stroke_accel_x_max = single_stroke["Accel_X"].max()
        Accel_X_max.append(single_stroke_accel_x_max)

        # Accel_Y_mean
        single_stroke_accel_y_mean = single_stroke["Accel_Y"].mean()
        Accel_Y_mean.append(single_stroke_accel_y_mean)

        # Accel_Y_min
        single_stroke_accel_y_min = single_stroke["Accel_Y"].min()
        Accel_Y_min.append(single_stroke_accel_y_min)

        # Accel_Y_max
        single_stroke_accel_y_max = single_stroke["Accel_Y"].max()
        Accel_Y_max.append(single_stroke_accel_y_max)

        # Accel_Z_mean
        single_stroke_accel_z_mean = single_stroke["Accel_Z"].mean()
        Accel_Z_mean.append(single_stroke_accel_z_mean)

        # Accel_Z_min
        single_stroke_accel_z_min = single_stroke["Accel_Z"].min()
        Accel_Z_min.append(single_stroke_accel_z_min)

        # Accel_Z_max
        single_stroke_accel_z_max = single_stroke["Accel_Z"].max()
        Accel_Z_max.append(single_stroke_accel_z_max)

    # Create new data frame
    feature_dataframe = pd.DataFrame({
        'Magn_X_mean': Magn_X_mean, 'Magn_X_min': Magn_X_min, 'Magn_X_max': Magn_X_max,
        'Magn_Y_mean': Magn_Y_mean, 'Magn_Y_min': Magn_Y_min, 'Magn_Y_max': Magn_Y_max,
        'Magn_Z_mean': Magn_Z_mean, 'Magn_Z_min': Magn_Z_min, 'Magn_Z_max': Magn_Z_max,
        'Gyro_X_mean': Gyro_X_mean, 'Gyro_X_min': Gyro_X_min, 'Gyro_X_max': Gyro_X_max,
        'Gyro_Y_mean': Gyro_Y_mean, 'Gyro_Y_min': Gyro_Y_min, 'Gyro_Y_max': Gyro_Y_max,
        'Gyro_Z_mean': Gyro_Z_mean, 'Gyro_Z_min': Gyro_Z_min, 'Gyro_Z_max': Gyro_Z_max,
        'Accel_X_mean': Accel_X_mean, 'Accel_X_min': Accel_X_min, 'Accel_X_max': Accel_X_max,
        'Accel_Y_mean': Accel_Y_mean, 'Accel_Y_min': Accel_Y_min, 'Accel_Y_max': Accel_Y_max,
        'Accel_Z_mean': Accel_Z_mean, 'Accel_Z_min': Accel_Z_min, 'Accel_Z_max': Accel_Z_max,
    })

    # Add condition to data frame
    if condition == condition_0:
        feature_dataframe['condition'] = 0
    elif condition == condition_1:
        feature_dataframe['condition'] = 1
    elif condition == condition_2:
        print('Daten für Vorhersage bereit!')
    else:
        print('Daten wurden nicht gelabelt. Kein Modell-Training möglich. Nur Vorhersage!')
    return feature_dataframe

if __name__ == '__main__':

    # Zwei Varianten: Train_model oder Make_prediction

    print('Was wollen Sie tun?\n0: Modell trainieren\n1: Technik vorhersagen')
    user_input = input()

    # Prepare date for Train_model:
    if user_input == '0':

        # Load data
        user_input_2 = input("Geben Sie den Ordnerpfad an, der alle Sensordaten (accel, gyro, magn) der Bedingung 'deep catch' beinhaltet: ")
        assert os.path.exists(user_input_2), "Ich konnte die Daten in folgendem Pfad nicht finden: " + str(user_input_2)
        raw_accel_0 = dd.read_csv(str(user_input_2)+'/accel-*.csv', header=None).compute()
        raw_gyro_0 = dd.read_csv(str(user_input_2)+'/gyro-*.csv', header=None).compute()
        raw_magn_0 = dd.read_csv(str(user_input_2)+'/magn-*.csv', header=None).compute()
        print("Sehr gut, es wurden alle Sensordaten der Bedingung 'deep catch' gefunden!")

        user_input_3 = input("Geben Sie den Ordnerpfad an, der alle Sensordaten (accel, gyro, magn) der Bedingung 'high elbow catch' beinhaltet: ")
        assert os.path.exists(user_input_3), "Ich konnte die Daten in folgendem Pfad nicht finden: " + str(user_input_3)
        raw_accel_1 = dd.read_csv(str(user_input_3) + '/accel-*.csv', header=None).compute()
        raw_gyro_1 = dd.read_csv(str(user_input_3) + '/gyro-*.csv', header=None).compute()
        raw_magn_1 = dd.read_csv(str(user_input_3) + '/magn-*.csv', header=None).compute()
        print('Hooray, wir haben alle deine Sensordaten gefunden!')

        # Create data frame with all sensor information
        column_names = ['Accel_X', 'Accel_Y', 'Accel_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z', 'Magn_X', 'Magn_Y', 'Magn_Z']
        raw_data_0 = pd.concat([raw_accel_0, raw_gyro_0, raw_magn_0], axis=1, sort=False)
        raw_data_0.columns = column_names

        raw_data_1 = pd.concat([raw_accel_1, raw_gyro_1, raw_magn_1], axis=1, sort=False)
        raw_data_1.columns = column_names

        # Create data frame for every single stroke and save as csv
        user_input_4 = input('Sollen die Schwimmzüge einzeln abgespeichert werden?\n0 Nein\n1 Ja\n')
        if user_input_4 == '0':
            single_stroke(raw_data_0, user_input_2, save_data=False)
            single_stroke(raw_data_1, user_input_3, save_data=False)
        elif user_input_4 == '1':
            single_stroke(raw_data_0, user_input_2, save_data=True)
            single_stroke(raw_data_1, user_input_3, save_data=True)
        else:
            print('Sie müssen entweder 0 (Nein) oder 1 (Ja) eingeben.')

        condition_0 = str(user_input_2)+'/stroke_*.csv'
        condition_1 = str(user_input_3)+'/stroke_*.csv'

        # Feature engineering
        feature_df_0 = feature_dataframe(condition_0)
        feature_df_1 = feature_dataframe(condition_1)
        frames = [feature_df_0, feature_df_1]
        feature_df_clean = pd.concat(frames)

        # Save clean_data to CSV
        pd.DataFrame(feature_df_clean).to_csv('/Users/marcsauer/PycharmProjects/Swim/data/clean_data.csv', index=False)
        print("'clean_data' als CSV abgespeichert.")
        print('Datenvorbereitung abgeschlossen!')

        # Train model
        print("Soll das Modell nun trainiert werden?\n0: Nein\n1: Ja")
        user_input_5 = input()

        if user_input_5 == '1':
            print('Das Modell wird nun trainiert!')

            # Read clean data
            clean_data = pd.read_csv('/Users/marcsauer/PycharmProjects/Swim/data/clean_data.csv')

            # Prepare data
            X = clean_data.drop(['condition'], axis=1)
            y = clean_data['condition']

            # Split data in train, validate and test set
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
            X_train = pd.DataFrame(X_train)
            X_val = pd.DataFrame(X_val)

            # Run training
            logreg = LogisticRegression(solver='liblinear', multi_class='ovr')
            model_logreg = logreg.fit(X_train, y_train)
            print('Training abgeschlossen!')

            # Evaluate training
            score_logreg = model_logreg.score(X_val, y_val)
            print('Die Genauigkeit des Modells liegt bei: {:.2f}%'.format(score_logreg * 100))

            # Save model
            print('Soll das Modell gespeichert werden?\n0: Nein\n1: Ja')
            user_input_6 = input()

            if user_input_6 == '1':
                print('Das Modell wird nun gespeichert!')

                # save the model to disk
                filename = 'trained_model.sav'
                pickle.dump(model_logreg, open(filename, 'wb'))
                print('Speichern des Modells abgeschlossen!')

            elif user_input_6 == '0':
                print('Soll ein weiteres Modell trainiert werden?')
                # ToDo:
                pass


        elif user_input_5 == '0':
            pass

    # Prepare data for making prediction:
    elif user_input == '1':

        # Load data
        user_input_5 = input('Geben Sie den Ordnerpfad an, der alle Sensordaten (accel, gyro, magn) beinhaltet: ')
        assert os.path.exists(user_input_5), "Ich konnte die Daten in folgendem Pfad nicht finden: " + str(user_input_5)
        raw_accel_2 = dd.read_csv(str(user_input_5)+'/accel-*.csv', header=None).compute()
        raw_gyro_2 = dd.read_csv(str(user_input_5)+'/gyro-*.csv', header=None).compute()
        raw_magn_2 = dd.read_csv(str(user_input_5)+'/magn-*.csv', header=None).compute()
        print('Sehr gut, es wurden alle Sensordaten gefunden!')

        # Create data frame with all sensor information
        column_names = ['Accel_X', 'Accel_Y', 'Accel_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z', 'Magn_X', 'Magn_Y', 'Magn_Z']
        raw_data_2 = pd.concat([raw_accel_2, raw_gyro_2, raw_magn_2], axis=1, sort=False)
        raw_data_2.columns = column_names

        # Create data frame for every single stroke and save as csv
        clean_data = single_stroke(raw_data_2, user_input_5, save_data=True)

        # ToDo: das ist nicht so schön, aber funktioniert. Noch anpassen.
        condition_0 = 'irgendwas'
        condition_1 = 'irgendwas'
        condition_2 = str(user_input_5) + '/stroke_*.csv'

        # Feature engineering
        clean_data_2 = feature_dataframe(condition_2)
        pd.DataFrame(clean_data_2)

        # Prepare data
        X = clean_data_2

        # Make prediction
        # Load the model from disk
        filename = 'trained_model.sav'
        loaded_model = pickle.load(open(filename, 'rb'))
        y_pred = loaded_model.predict(X)
        print(y_pred)


    else:
        print("Falsche Eingabe. Wählen Sie entweder 0 (Modell trainieren) oder 1 (Technik vorhersagen).")
        user_input = input()