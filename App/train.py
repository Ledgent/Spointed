import pandas as pd
from lightgbm import LGBMRegressor, Dataset, cv
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
df = pd.read_csv('static/data/locations.csv')

# Encode categorical features
label_encoders = {}
categorical_columns = ['type', 'business_size', 'state', 'target_audience', 'foot_traffic', 'affordability', 'competitors']

for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Define features and target
X = df.drop(columns=['name', 'budget'])
y = df['budget']

# Normalize numerical features
scaler = StandardScaler()
X[['foot_traffic', 'affordability']] = scaler.fit_transform(X[['foot_traffic', 'affordability']])

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Prepare LightGBM dataset
lgb_train = Dataset(X_train, label=y_train)
lgb_eval = Dataset(X_test, label=y_test, reference=lgb_train)

# Set parameters for LightGBM
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.05,
    'max_depth': 5,
    'num_leaves': 31,
    'n_estimators': 200,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'verbose': -1
}

# Train with built-in cross-validation and early stopping
cv_results = cv(params, lgb_train, nfold=3, early_stopping_rounds=10, verbose_eval=True)

# Train the final model
best_iter = len(cv_results['rmse-mean'])
final_model = LGBMRegressor(
    learning_rate=params['learning_rate'],
    max_depth=params['max_depth'],
    num_leaves=params['num_leaves'],
    n_estimators=best_iter,
    subsample=params['subsample'],
    colsample_bytree=params['colsample_bytree'],
    random_state=42
)
final_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10, verbose=10)

# Evaluate the model
y_pred = final_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')
print(f'R2 Score: {r2:.2f}')

# Save the trained model and label encoders
joblib.dump(final_model, 'location_recommendation_model.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(scaler, 'scaler.pkl')
