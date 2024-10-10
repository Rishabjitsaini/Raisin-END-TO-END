
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load(r'C:\Users\sari3003\Data science\test\raisin_model.pkl')

@app.route('/')
def home():
    return 'Raisin Type Prediction Model'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the request
        data = request.get_json(force=True)
        
        # Ensure all required features are provided
        features = ['Area', 'MajorAxisLength', 'MinorAxisLength', 'Eccentricity', 'ConvexArea', 'Extent', 'Perimeter']
        input_data = [data[feature] for feature in features]
        
        # Convert input data to numpy array and reshape for prediction
        input_data = np.array(input_data).reshape(1, -1)
        
        # Make prediction using the model
        prediction = model.predict(input_data)
        
        # Return the predicted raisin type (Kecimen or Besni)
        raisin_type = 'Kecimen' if prediction[0] == 0 else 'Besni'
        return jsonify({'prediction': raisin_type})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
