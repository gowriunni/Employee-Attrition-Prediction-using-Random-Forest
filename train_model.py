
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Load the dataset
df = pd.read_csv(
    "WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

# 2. Select only the employee details we need
selected_features = [
    "Age",
    "Department",
    "JobRole",
    "MonthlyIncome",
    "JobSatisfaction",
    "OverTime"
]

X = df[selected_features]
y = df["Attrition"]

# 3. Identify numerical and categorical columns
numerical_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_columns = X.select_dtypes(
    include=["object", "category", "bool", "str"]
).columns.tolist()

# 4. Prepare numerical data
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

# 5. Prepare categorical data
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# 6. Combine preprocessing steps
preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_columns),
    ("cat", categorical_pipeline, categorical_columns)
])

# 7. Create the Random Forest model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ))
])

# 8. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Total records:", len(df))
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Selected features:", selected_features)

# 9. Train the model
model.fit(X_train, y_train)

print("\nModel training completed!")

# 10. Make predictions on test data
y_pred = model.predict(X_test)

# 11. Evaluate the model
print(
    "\nAccuracy:",
    round(accuracy_score(y_test, y_pred), 4)
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 12. Save the trained model
joblib.dump(model, "attrition_model.joblib")

print("\nModel saved successfully!")
print("Saved as: attrition_model.joblib")
