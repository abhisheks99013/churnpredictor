import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

# Read the data
df = pd.read_csv("telco_churn.csv")

# Convert 'TotalCharges' column to numeric values and fill missing values with the median
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Drop the 'customerID' column as it is not useful for the model
df.drop('customerID', axis=1, inplace=True)

# Convert 'Churn' column to binary values: 'Yes' = 1, 'No' = 0
df['Churn'] = df['Churn'].apply(lambda x: 1 if x in ['True', 'Yes'] else 0)

# Encode categorical variables using LabelEncoder
le = LabelEncoder()
for column in df.select_dtypes(include='object').columns:
    df[column] = le.fit_transform(df[column])

# Split data into features (X) and target (y)
X = df.drop('Churn', axis=1)
y = df['Churn']

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize the feature data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Random Forest model with balanced class weights
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)

# Predict on the test set
y_pred = rf.predict(X_test)

# Print the classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix visualization
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Feature importance visualization
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
plt.title('Top 10 Features Driving Churn')
plt.show()

# Print top 10 important features
print(feature_importance.head(10))

# Apply SMOTE to balance the training data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Re-train the model with SMOTE data
rf_smote = RandomForestClassifier(n_estimators=100, random_state=42)
rf_smote.fit(X_train_smote, y_train_smote)

# Predict on the test set again with SMOTE data
y_pred_smote = rf_smote.predict(X_test)

# Print the classification report after SMOTE
print("Classification Report (SMOTE):")
print(classification_report(y_test, y_pred_smote))

# Calculate the savings from reduced churn
churn_rate = y_test.mean()
reduced_churn_rate = churn_rate * 0.85  # Assuming 15% reduction in churn
avg_revenue_per_customer = 70  # Example average revenue per customer
n_customers = len(y_test)
savings = (churn_rate - reduced_churn_rate) * n_customers * avg_revenue_per_customer
print(f"Estimated monthly savings from 15% churn reduction: ${savings:.2f}")
