import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from feature_engine.encoding import OneHotEncoder
from feature_engine.transformation import PowerTransformer

data = pd.read_csv('data.csv')

print("Original Data:")
print(data.head())

data['Date'] = pd.to_datetime(data['Date'])
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day

data['Interaction_feature'] = data['Feature1'] * data['Feature2']
data['Squared_feature'] = data['Feature1'] ** 2

X = data.drop(['Target', 'Date'], axis=1)
y = data['Target']

constant_features = X.columns[X.nunique() == 1]
X = X.drop(constant_features, axis=1)

encoder = OneHotEncoder(variables=['Cat1', 'Cat2'], drop_last=True)
X_encoded = encoder.fit_transform(X)

transformer = PowerTransformer(variables=['Feature1', 'Feature2'])
X_transformed = transformer.fit_transform(X_encoded)

X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

# Check for constant features in X_train after transformation
constant_features_after_transform = X_train.columns[X_train.nunique() == 1]
X_train = X_train.drop(constant_features_after_transform, axis=1)

X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
X_train.dropna(axis=1, inplace=True)

selector = SelectKBest(score_func=f_classif, k=5)
X_train_selected = selector.fit_transform(X_train, y_train)

selected_features = X_train.columns[selector.get_support(indices=True)]

print("\nSelected Features:")
print(selected_features)
