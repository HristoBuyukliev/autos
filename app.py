from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load your scikit-learn model
model = joblib.load('best_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Make predictions using your model
    predictions = model.predict(pd.DataFrame(data))

    return jsonify({'prediction': predictions[0]})

if __name__ == '__main__':
    app.run(debug=True)
