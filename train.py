import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from clean import *
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, KFold
import joblib


data = pd.read_csv('Automobile_data.csv')
data = data[data.price != '?']

# define missing values filler
mf = MissingFiller()

# Define preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), mf.cat_columns)
    ], remainder='passthrough'
)

pipeline = Pipeline([
    ('fill_missing', mf),
    ('dummifier', preprocessor),
    ('scaler', StandardScaler()),
    ('model', GradientBoostingRegressor())
])

# Define the grid of hyperparameters to search
param_grid = {
    'model__n_estimators': [500, 1000, 1500],
    'model__learning_rate': [0.01, 0.1, 0.2],
    'model__max_depth': [3, 4, 5]
}

X = data[mf.cat_columns+mf.cont_columns]
y = data['price']

# Setup the grid search
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_absolute_error')

# Perform nested cross-validation, and output the average score
kfold = KFold(n_splits=5, shuffle=True, random_state=1)
nested_score = cross_val_score(grid_search, data[mf.cat_columns+mf.cont_columns], 
                               y=data['price'].astype('float'), cv=kfold,scoring='neg_mean_absolute_error')

print(f'Nested cross-validation scores: {nested_score}')
print(f'Average nested cross-validation score: {np.mean(nested_score)}')

grid_search.fit(data[mf.cat_columns+mf.cont_columns], y=data['price'].astype('float'))
best_params = grid_search.best_params_
print(f'Best parameters from the nested cross validation: {best_params}')

# Save the model as a pickle file
joblib.dump(grid_search.best_estimator_, 'best_model.pkl')

print("Model saved successfully!")

# load model
# Load the model from the pickle file
loaded_model = joblib.load('best_model.pkl')

print("Model loaded successfully!")
