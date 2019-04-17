import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Train model

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
user_input = input()

if user_input == '1':
    print('Das Modell wird nun gespeichert!')
    # save the model to disk
    filename = 'trained_model.sav'
    pickle.dump(model_logreg, open(filename, 'wb'))
    print('Speichern des Modells abgeschlossen!')
elif user_input == '0':
    print('Soll ein weiteres Modell trainiert werden?')
    # ToDo:
    pass
if user_input == '0':
    pass