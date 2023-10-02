import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


cat_columns = ['symboling', 'make', 'fuel-type', 'aspiration', 
                            'num-of-doors', 'body-style', 'drive-wheels', 
                            'engine-location', 'engine-type', 'num-of-cylinders', 
                            'fuel-system']

cont_columns = ['normalized-losses', 'wheel-base', 'length', 'width', 
                        'height', 'curb-weight', 'engine-size', 'bore', 'stroke', 
                        'peak-rpm', 'city-mpg', 'highway-mpg', 
                        'horsepower', 'compression-ratio']

def dummify(data):
    data = data[cat_columns + cont_columns + ['price']]
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_columns)
        ], remainder='passthrough'
    )
    return preprocessor.fit_transform(data)

class MissingFiller:
    def __init__(self):
        self.filler_dict = {}
        # fill all columns
        self.cont_columns = cont_columns
        self.cat_columns = cat_columns

    def fit(self, data, labels=None):
        new_data = data.copy()
        for col in data.columns:
            if col == 'price': continue
            if col not in self.cont_columns:
                new_data, mean = self.categorical_mean_fill(data, col)
                self.filler_dict[col] = mean
            if col in self.cont_columns:
                new_data, mode = self.continuous_mean_fill(data, col)
                self.filler_dict[col] = mode
        return new_data
    
    def transform(self, data, labels=None):
        for col in data.columns:
            if col not in self.filler_dict: continue
            data.loc[data[col] == '?', col] = self.filler_dict[col]
            if col in self.cont_columns:
                data[col] = data[col].astype('float')
        return data
    
    def fit_transform(self, data, labels):
        return self.fit(data, labels)

    def continuous_mean_fill(self, data, col):
        mean = data[data[col] != '?'][col].astype('float').mean()
        data.loc[data[col] == '?', col] = mean
        data[col] = data[col].astype(float)
        return data, mean

    def categorical_mean_fill(self, data, col):
        mode = data[data[col] != '?'][col].mode()[0]
        data.loc[data[col] == '?', col] = mode
        return data, mode