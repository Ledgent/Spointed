import joblib

def load_model():
    try:
        model = joblib.load('location_recommendation_model.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        return model, label_encoders
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None
