import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Read clean data
clean_data = pd.read_csv('/Users/marcsauer/PycharmProjects/Swim/data/clean_data1.csv')

# Prepare data
X = clean_data[['Magn_Y_mean', 'Magn_Y_min', 'Magn_Y_max', 'Magn_Z_mean']]
y = clean_data['condition']

# Split data in train, validate and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
X_train = pd.DataFrame(X_train)
X_val = pd.DataFrame(X_val)

# Run training
logreg = LogisticRegression(solver='liblinear', multi_class='ovr')
model_logreg = logreg.fit(X_train, y_train)

# Evaluate training
score_logreg = model_logreg.score(X_val, y_val)
print('Score Logistic Regression with 4 features: {:.2f}%'.format(score_logreg*100))
