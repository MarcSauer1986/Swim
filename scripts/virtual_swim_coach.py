import pandas as pd
import numpy as np
import glob
from scipy.signal import find_peaks
import subprocess

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
            pd.DataFrame(list_of_dfs_without_outliers[i]).to_csv('{}/stroke_{}.csv'.format(user_input_2, k), index=False)
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

    # 3 Options: Preprocess data, Train model or Make prediction
    print('Was wollen Sie tun?\n0: Daten vorbereiten\n1: Modell trainieren\n2: Technik vorhersagen')
    user_input = input()

    # Preprocess data:
    if user_input == '0':
        subprocess.call(['python', 'preprocess_data.py'])

    # Train model:
    if user_input == '1':
        subprocess.call(['python', 'train_model.py'])

    # Make prediction:
    if user_input == '2':
        subprocess.call(['python', 'make_prediction.py'])

    else:
        print("Falsche Eingabe. Wählen Sie entweder 0 (Daten vorbereiten), 1 (Modell trainieren) oder 2 (Technik vorhersagen).")
        user_input = input()