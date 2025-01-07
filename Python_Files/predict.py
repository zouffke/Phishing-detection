# Imports
import sys
import joblib
from analyse_url import URLAnalyzer
from custom_scaler import scale

# Load the model
model = joblib.load('../models/random_forest.pkl')

# Load the scaler
scaler = joblib.load('../models/scaler.pkl')

# Analyze the URL
analyser = URLAnalyzer()
features = analyser.analyse_url(sys.argv[1])

# Use custom scaler
features = scale(features)

# Scale the features
features_scaled = scaler.transform(features)

# Predict using the model
prediction = model.predict(features_scaled)
prediction_proba = model.predict_proba(features_scaled)
print(f'Prediction: {prediction[0]}')
print(f"Certainty: {max(prediction_proba[0]) * 100:.2f}%")
