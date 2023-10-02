import requests

sample_data = [{'symboling': 3, 'normalized-losses': 122.0, 'make': 'alfa-romero', 
        'fuel-type': 'gas', 'aspiration': 'std', 'num-of-doors': 'two', 
        'body-style': 'convertible', 'drive-wheels': 'rwd', 'engine-location': 'front', 
        'wheel-base': 88.6, 'length': 168.8, 'width': 64.1, 'height': 48.8, 
        'curb-weight': 2548.0, 'engine-type': 'dohc', 'num-of-cylinders': 'four', 
        'engine-size': 130.0, 'fuel-system': 'mpfi', 'bore': 3.47, 
        'stroke': 2.68, 'compression-ratio': 9.0, 'horsepower': 111.0, 
        'peak-rpm': 5000.0, 'city-mpg': 21.0, 'highway-mpg': 27.0, 'price': '13495'}]  # Replace with your feature values
response = requests.post('http://127.0.0.1:5000/predict', json=sample_data)

print(response.json()) 