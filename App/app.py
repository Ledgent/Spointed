from flask import Flask, render_template, request
from model import load_model  # Load the model loader function
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('recommendation.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    # Fetch user input from the form
    company_name = request.form['company_name']
    restaurant_type = request.form['restaurant_type']
    budget = request.form['budget']
    state = request.form['state']
    business_size = request.form['business_size']

    # Convert budget to float and handle potential errors
    try:
        budget = float(budget)
    except ValueError:
        return "Error: Budget must be a numeric value."

    # Load the trained model and label encoders
    model, label_encoders = load_model()
    if not model:
        return "Error: Model not loaded. Please ensure the model file is available."

    # Encode the input features using label encoders
    restaurant_type_encoded = label_encoders['restaurant_type'].transform([restaurant_type])[0]
    state_encoded = label_encoders['state'].transform([state])[0]
    business_size_encoded = label_encoders['business_size'].transform([business_size])[0]

    # Prepare the input features for prediction (adjust feature order as needed)
    feature_data = np.array([
        restaurant_type_encoded,
        business_size_encoded,
        state_encoded,
        budget
    ]).reshape(1, -1)

    # Predict and format the recommendation
    try:
        recommendation = model.predict(feature_data)[0]
    except Exception as e:
        recommendation = f"Error during prediction: {e}"

    # Render results.html with the recommendation and user details
    return render_template(
        'results.html',
        company_name=company_name,
        restaurant_type=restaurant_type,
        budget=budget,
        state=state,
        business_size=business_size,
        recommendation=recommendation
    )

if __name__ == "__main__":
    app.run(debug=True)
