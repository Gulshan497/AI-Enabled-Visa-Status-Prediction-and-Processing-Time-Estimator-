# IMPORT PACKAGES
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

pd.set_option("display.max_columns", None)

# PART 1: LARGE VISA DATASET
print("\n===== PART 1: LARGE VISA DATASET =====\n")

data = {
    "application_date": [
        "2024-01-01", "2024-02-15", "2024-03-10",
        "2024-04-05", "2024-05-12", "2024-06-01",
        "2024-06-18", "2024-07-02", "2024-07-25",
        "2024-08-10"
    ],
    "decision_date": [
        "2024-02-01", "2024-03-20", "2024-04-05",
        "2024-05-01", "2024-06-20", "2024-06-25",
        "2024-07-28", "2024-08-05", "2024-09-15",
        "2024-10-01"
    ],
    "country": [
        "India", "United States", "United Kingdom",
        "Canada", "Australia", "Germany",
        "India", "France", "Japan", "Brazil"
    ],
    "visa_type": [
        "Student", "Tourist", "Work",
        "Tourist", "Student", "Work",
        "Work", "Tourist", "Student", "Work"
    ]
}

df = pd.DataFrame(data)
print("Original DataFrame:\n", df)


# DATE CONVERSION
df["application_date"] = pd.to_datetime(df["application_date"])
df["decision_date"] = pd.to_datetime(df["decision_date"])

print("\nAfter Date Conversion:\n", df)

# PROCESSING DAYS
df["processing_days"] = (
    df["decision_date"] - df["application_date"]
).dt.days

print("\nAfter Calculating Processing Days:\n", df)

# ENCODING
df_encoded = pd.get_dummies(df, columns=["country", "visa_type"])

print("\nEncoded DataFrame:\n", df_encoded)

# MACHINE LEARNING
X = df_encoded.drop(
    columns=["processing_days", "application_date", "decision_date"]
)
y = df_encoded["processing_days"]

model = LinearRegression()
model.fit(X, y)


# PREDICTION SAMPLE
sample_input = pd.DataFrame(
    np.zeros((1, len(X.columns))),
    columns=X.columns
)

# Example: India + Student Visa
sample_input.loc[0, "country_India"] = 1
sample_input.loc[0, "visa_type_Student"] = 1

predicted_days = model.predict(sample_input)

print("\nPredicted Processing Time (India + Student):",
      round(predicted_days[0], 2), "days")
