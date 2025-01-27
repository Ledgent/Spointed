from flask import Blueprint, request, render_template
import numpy as np
from model import load_model  # Import the load_model function

# Create a Blueprint for the recommend route
recommend = Blueprint('recommend', __name__)

@recommend.route('/recommend', methods=['POST'])
def recommend_location():
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
    if not model or not label_encoders:
        return "Error: Model or label encoders not loaded. Please check your setup."

    # Encode the input features using the label encoders
    try:
        restaurant_type_encoded = label_encoders['type'].transform([restaurant_type])[0]
        state_encoded = label_encoders['state'].transform([state])[0]
        business_size_encoded = label_encoders['business_size'].transform([business_size])[0]
    except KeyError as e:
        return f"Error: Missing encoder for {e}"

    # Prepare the input features for prediction
    feature_data = np.array([
        restaurant_type_encoded,
        business_size_encoded,
        state_encoded,
        budget
    ]).reshape(1, -1)

    # Predict the location recommendation using the model
    try:
        recommendation = model.predict(feature_data)[0]
    except Exception as e:
        return f"Error during prediction: {e}"

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