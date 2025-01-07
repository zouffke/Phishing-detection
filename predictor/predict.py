# Imports
import sys
import joblib
from resources.scripts.analyse_url import URLAnalyzer
from resources.scripts.custom_scaler import scale
import pandas as pd

# Load the model
model = joblib.load('resources/random_forest.pkl')

# Load the scaler
scaler = joblib.load('resources/scaling/scaler.pkl')

# Analyze the URL
analyser = URLAnalyzer()
features = analyser.analyse_url(sys.argv[1])

# Use custom scaler
features = scale(features, pd.read_csv('resources/scaling/scale.csv', header=0, index_col=0))

# Scale the features
features_scaled = scaler.transform(features)

# Predict using the model
prediction = model.predict(features_scaled)
prediction_proba = model.predict_proba(features_scaled)
print(f'Prediction: {prediction[0]}')
print(f"Certainty: {max(prediction_proba[0]) * 100:.2f}%")
