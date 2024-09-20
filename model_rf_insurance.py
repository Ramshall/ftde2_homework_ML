
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Load dataset
data = pd.read_csv('./data/Homework - Prediction Insurance.csv')

# Preprocessing
le_gender = LabelEncoder()
le_vehicle_age = LabelEncoder()
le_vehicle_damage = LabelEncoder()

data['Gender'] = le_gender.fit_transform(data['Gender'])
data['Vehicle_Age'] = le_vehicle_age.fit_transform(data['Vehicle_Age'])
data['Vehicle_Damage'] = le_vehicle_damage.fit_transform(data['Vehicle_Damage'])

# Define features and target
X = data.drop(columns=['id', 'Response'])  # Drop id
y = data['Response']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Predict on the test set
y_pred_rf = rf_model.predict(X_test)

# Evaluate the model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
class_report_rf = classification_report(y_test, y_pred_rf)

# Print evaluation metrics
print(f"Accuracy: {accuracy_rf}")
print("Confusion Matrix:")
print(conf_matrix_rf)
print("Classification Report:")
print(class_report_rf)

# Save the model
joblib.dump(rf_model, './model/model_insurance.pkl')
print("Model saved as model_insurance.pkl")
