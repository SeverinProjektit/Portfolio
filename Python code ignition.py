import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc

# Generate synthetic wildfire ignition data
np.random.seed(42)
num_samples = 500

# Simulated environmental variables
temperature = np.random.uniform(10, 40, num_samples)  # Temperature in Celsius
humidity = np.random.uniform(10, 90, num_samples)  # Humidity in percentage
wind_speed = np.random.uniform(0, 20, num_samples)  # Wind speed in m/s

# Define true beta coefficients for logistic model
beta_0 = -5    # Baseline probability shift
beta_1 = 0.15  # Effect of temperature
beta_2 = -0.1  # Effect of humidity (negative as humidity suppresses fire)
beta_3 = 0.2   # Effect of wind speed

# Compute logistic probability
logit = beta_0 + beta_1 * temperature + beta_2 * humidity + beta_3 * wind_speed
probability = 1 / (1 + np.exp(-logit))  # Logistic function

# Generate binary ignition labels (1 = fire ignites, 0 = no fire) based on probability
ignition = np.random.binomial(1, probability)

# Create DataFrame
df = pd.DataFrame({'Temperature': temperature, 'Humidity': humidity, 'Wind Speed': wind_speed, 'Ignition': ignition})

# Split data into training and testing sets
X = df[['Temperature', 'Humidity', 'Wind Speed']]
y = df['Ignition']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict ignition probabilities
y_pred_prob = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

# Compute model accuracy
accuracy = accuracy_score(y_test, y_pred)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Wildfire Ignition Prediction')
plt.legend(loc='lower right')
plt.show()

# Display the first few rows of the dataset and the model accuracy
import ace_tools as tools
tools.display_dataframe_to_user(name="Wildfire Ignition Data", dataframe=df)

accuracy
