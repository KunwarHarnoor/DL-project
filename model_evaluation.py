import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Step 1: Load the dataset
data = pd.read_csv('your_dataset.csv')  # Assuming the data is in a CSV file
print("Data Loaded Successfully")

# Step 2: Preprocessing
# Assuming you have columns like 'humidity', 'temperature', 'windspeed', and 'AQI'
features = data[['humidity', 'temperature', 'wind_speed']]  # Features
target = data['AQI']  # Target variable (AQI)

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Step 4: Train a Regression Model
# RandomForestRegressor for continuous AQI prediction
reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
reg_model.fit(X_train, y_train)

# Step 5: Make Predictions (Regression)
y_pred_reg = reg_model.predict(X_test)

# Step 6: Performance Evaluation (Regression)
# 6.1: Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred_reg)
print(f'Mean Absolute Error (MAE): {mae:.2f}')

# 6.2: R-Squared (R²)
r2 = r2_score(y_test, y_pred_reg)
print(f'R-Squared (R²): {r2:.2f}')

# 6.3: Root Mean Squared Error (RMSE)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_reg))
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')

# 6.4: Plotting Actual vs Predicted AQI (Regression)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_reg)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', lw=2)
plt.xlabel('Actual AQI')
plt.ylabel('Predicted AQI')
plt.title('Actual vs Predicted AQI (Regression)')
plt.show()

# Step 7: Convert AQI into categories for classification (Optional)
bins = [0, 50, 100, 150, 200, 300, 500]
labels = ['Good', 'Moderate', 'Unhealthy for Sensitive Groups', 'Unhealthy', 'Very Unhealthy', 'Hazardous']
data['AQI_Category'] = pd.cut(data['AQI'], bins=bins, labels=labels)

# Step 8: Prepare data for classification
target_class = data['AQI_Category']  # Categorical target variable now

# Split again for classification task
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(features, target_class, test_size=0.2, random_state=42)

# Step 9: Train a Classification Model
# RandomForestClassifier for categorical AQI classification
class_model = RandomForestClassifier(n_estimators=100, random_state=42)
class_model.fit(X_train_class, y_train_class)

# Step 10: Make Predictions (Classification)
y_pred_class = class_model.predict(X_test_class)

# Step 11: Performance Evaluation (Classification)
# 11.1: Accuracy
accuracy = accuracy_score(y_test_class, y_pred_class)
print(f'Accuracy: {accuracy:.2f}')

# 11.2: Precision
precision = precision_score(y_test_class, y_pred_class, average='macro', labels=class_model.classes_,zero_division=1)
print(f'Precision: {precision:.2f}')

# 11.3: Recall
recall = recall_score(y_test_class, y_pred_class, average='macro', labels=class_model.classes_,zero_division=1)
print(f'Recall: {recall:.2f}')



print("Models saved successfully")

