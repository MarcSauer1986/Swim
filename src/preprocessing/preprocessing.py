import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


class Preprocessing:
    def __init__(self, name):
        self.name = name.lower()
        self.data = {}

        root_dir = os.path.dirname(__file__)
        directory_template = '{root_dir}/../../data/{name}'
        self.directory = directory_template.format(root_dir=root_dir, name=name)

        if not os.path.exists(self.directory):
            print(f'Creating "{name}" directory for you!')
            os.makedirs(self.directory)

    def load_data(self, filename, filetype='csv', *, name, **kwargs):
        filepath = f'{self.directory}/{filename}'

        function_name = f'read_{filetype}'
        df = getattr(pd, function_name)(filepath, **kwargs)
        self.data[name] = df
        return df

    def save(self, name, filetype='csv', *, index=False, **kwargs):
        filepath = f'{self.directory}/{name}.{filetype}'
        getattr(self.data[name], f'to_{filetype}')(filepath, index=index, **kwargs)

    def cleanup(self, name, *, drop=None, drop_duplicates=False, dropna=None):
        data = self.data[name]

        if drop is not None:
            data = data.drop(columns=drop)

        if drop_duplicates is True:
            data = data.drop_duplicates()

        if dropna is not None:
            if 'axis' not in dropna:  # das ist ein Default setting... falls man es nicht angibt, wird axis=1 angenommen
                dropna['axis'] = 1
            data = data.dropna(**dropna)  # **dropna kann alle kwargs aufrufen von dropna funtion.
            # die optionen kann man als dictionary speichern und dann hier einlesen
            # zb : dropna_options = { 'axis' : 1,
            #  'thresh' : 2}
            # und dann: data.cleanup(dropna= dropna_options

        self.data['clean'] = data

    def label_encode(self, *, columns):
        if 'clean' not in self.data:
            print('Can not find clean data. Call .cleanup() first.')
            return

        data = self.data['clean']
        encoder = preprocessing.LabelEncoder()
        labels = pd.DataFrame()

        label_index = 0
        for column in columns:
            encoder.fit(data[column])
            label = encoder.transform(data[column])
            labels.insert(label_index, column=column, value=label)
            label_index += 1

        data = data.drop(columns, axis=1)
        data = pd.concat([data, labels], axis=1)
        self.data['clean'] = data

        return data

    def one_hot_encode(self, *, columns):
        if 'clean' not in self.data:
            print('Can not find clean data. Call .cleanup() first.')
            return

        data = self.data['clean']
        categorical = pd.get_dummies(data[columns], dtype='int')
        data = pd.concat([data, categorical], axis=1, sort=False)
        self.data['clean'] = data

        return data

    def get(self, name):
        return self.data[name]

    def set(self, name, value):
        self.data[name] = value

    def set_split(self, X, y, *, test_size=0.2, val_size=0.25, norm=True, random_state=42):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        if val_size is not False:
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=random_state)

            if norm is True:
                X_train_values = X_train.values  # returns a numpy array
                min_max_scaler = preprocessing.MinMaxScaler()
                X_train_scaled = min_max_scaler.fit_transform(X_train_values)
                X_train_standardized = pd.DataFrame(X_train_scaled, columns=X_train.columns)

                X_test_scaled = min_max_scaler.transform(X_test)
                X_test_standardized = pd.DataFrame(X_test_scaled, columns=X_train.columns)

                X_val_scaled = min_max_scaler.transform(X_val)
                X_val_standardized = pd.DataFrame(X_val_scaled, columns=X_train.columns)

                self.data['X_train'] = X_train_standardized
                self.data['X_test'] = X_test_standardized

                self.data['y_train'] = y_train
                self.data['y_test'] = y_test

                self.data['X_val'] = X_val_standardized
                self.data['y_val'] = y_val
                for k in ('X_train', 'y_train', 'X_test', 'y_test', 'X_val', 'y_val'):
                    self.save(name=k, index=False)
            else:
                self.data['X_train'] = X_train
                self.data['X_test'] = X_test

                self.data['y_train'] = y_train
                self.data['y_test'] = y_test

                self.data['X_val'] = X_val
                self.data['y_val'] = y_val
                for k in ('X_train', 'y_train', 'X_test', 'y_test', 'X_val', 'y_val'):
                    self.save(name=k, index=False)
        else:
            if norm is True:
                X_train_values = X_train.values  # returns a numpy array
                min_max_scaler = preprocessing.MinMaxScaler()
                X_train_scaled = min_max_scaler.fit_transform(X_train_values)
                X_train_standardized = pd.DataFrame(X_train_scaled, columns=X_train.columns)

                X_test_scaled = min_max_scaler.transform(X_test)
                X_test_standardized = pd.DataFrame(X_test_scaled, columns=X_train.columns)

                self.data['X_train'] = X_train_standardized
                self.data['X_test'] = X_test_standardized
                self.data['y_train'] = y_train
                self.data['y_test'] = y_test
                for k in ('X_train', 'y_train', 'X_test', 'y_test'):
                    self.save(name=k, index=False)
            else:
                self.data['X_train'] = X_train
                self.data['X_test'] = X_test

                self.data['y_train'] = y_train
                self.data['y_test'] = y_test
                for k in ('X_train', 'y_train', 'X_test', 'y_test'):
                    self.save(name=k, index=False)

        a = (1 - test_size) * 100

        if val_size is not False:
            b = a / (1 / val_size)
            c = 100.00 - (test_size * 100) - b

            print('{}%, {}%, {}% split for training, validation and test sets.'.format(c, b, (test_size) * 100))
        else:
            print('{}%, {}%, split for training and test sets.'.format(a, (test_size) * 100))
        if val_size is not False:
            return self.data['X_train'], self.data['y_train'], self.data['X_val'], self.data['y_val'],\
                   self.data['X_test'], self.data['y_test']
        else:
            return self.data['X_train'], self.data['y_train'], self.data['X_test'], self.data['y_test']
