from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load(r'C:\Users\sari3003\Data science\test\raisin_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the form
        area = float(request.form['Area'])
        major_axis_length = float(request.form['MajorAxisLength'])
        minor_axis_length = float(request.form['MinorAxisLength'])
        eccentricity = float(request.form['Eccentricity'])
        convex_area = float(request.form['ConvexArea'])
        extent = float(request.form['Extent'])
        perimeter = float(request.form['Perimeter'])
        
        # Prepare input data for the model
        input_data = np.array([[area, major_axis_length, minor_axis_length, eccentricity, convex_area, extent, perimeter]])
        
        # Make prediction using the model
        prediction = model.predict(input_data)
        
        # Return the predicted raisin type (Kecimen or Besni)
        raisin_type = 'Kecimen' if prediction[0] == 0 else 'Besni'
        return render_template('index.html', prediction=raisin_type)
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

