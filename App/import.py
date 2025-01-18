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

    # Convert budget to float and handle potential errors
    try:
        budget = float(budget)
    except ValueError:
        return "Error: Budget must be a numeric value."

    # Map restaurant_type to a numeric value (adjust to match model encoding)
    restaurant_type_encoded = 0 if restaurant_type.lower() == "traditional" else 1

    # Load the trained model
    model = load_model()
    if not model:
        return "Error: Model not loaded. Please ensure the model file is available."

    # Prepare the input features for prediction (adjust feature order as needed)
    # Example features: [restaurant_type, budget]
    feature_data = np.array([restaurant_type_encoded, budget]).reshape(1, -1)

    # Predict and format the recommendation
    try:
        recommendation = model.predict(feature_data)[0]
    except Exception as e:
        recommendation = f"Error during prediction: {e}"

    # Render display.html with the recommendation and user details
    return render_template(
        'results.html',
        company_name=company_name,
        restaurant_type=restaurant_type,
        budget=budget,
        state=state,
        recommendation=recommendation
    )

if __name__ == "__main__":
    app.run(debug=True)
