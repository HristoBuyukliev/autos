# autos

## 1. Installation
`pip install -r requirements.txt'

## 2. EDA
Take a look at the EDA.ipynb notebook. After each of the three sections, I've written a few conclusions. 

## 3. MCA
Take a look at the MCA.ipynb notebook. The main takeaways are that the data is not really explainable in two dimensions, 
and how the different categories are embedded. 

## 4. Modelling. 
You can train a model with `python train.py`. After some cleaning, we perform validation to discover the optimal parameters of a gradient boosted regressor, and nested cross-validation to get accurate estimates of it's loss. We use MAE as metric.    

## 5. Hosting a model. 
You can host a simple Flask server, that accepts HTTP requests, with `python app.py`. You can test if it works with `python test_app.py`. 