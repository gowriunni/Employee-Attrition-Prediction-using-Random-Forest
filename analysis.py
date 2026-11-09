
import pandas as pd


# Load the dataset
file_path = "WA_Fn-UseC_-HR-Employee-Attrition.csv"
df = pd.read_csv(file_path)

# 1. Display the first five employees
print("\n--- First 5 Employees ---")
print(df.head())

# 2. Check the number of rows and columns
print("\n--- Dataset Shape ---")
print(df.shape)

# 3. Display all column names
print("\n--- Column Names ---")
print(df.columns.tolist())

# 4. Check data types and missing values
print("\n--- Dataset Information ---")
df.info()

# 5. Count missing values in each column
print("\n--- Missing Values ---")
print(df.isnull().sum())

# 6. Check duplicate rows
print("\n--- Duplicate Rows ---")
print(df.duplicated().sum())

# 7. Count employees who left and stayed
print("\n--- Attrition Count ---")
print(df["Attrition"].value_counts())

# 8. Calculate attrition percentages
print("\n--- Attrition Percentage ---")
print(df["Attrition"].value_counts(normalize=True) * 100)


import matplotlib.pyplot as plt

attrition_counts = df["Attrition"].value_counts()

attrition_counts.plot(
    kind="bar",
    title="Employee Attrition Distribution",
    xlabel="Attrition",
    ylabel="Number of Employees",
    rot=0
)

plt.tight_layout()
plt.show()


# Separate the target from the input features
X = df.drop(columns=["Attrition"])
y = df["Attrition"]

print("\nNumber of input features:", X.shape[1])
print("Number of target values:", y.shape[0])

# Find numerical columns
numerical_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()

# Find categorical columns
categorical_columns = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("\nNumerical columns:")
print(numerical_columns)

print("\nCategorical columns:")
print(categorical_columns)

print("\nTarget values:")
print(y.unique())
