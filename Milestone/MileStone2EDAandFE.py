# IMPORT PACKAGES
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_columns", None)

# LOAD FULL VISA DATASET FROM CSV (visa_dataset.csv created in Milestone 1)
csv_path = os.path.join(os.path.dirname(__file__), "..", "visa_dataset.csv")
df = pd.read_csv(csv_path)
print("Original DataFrame loaded from visa_dataset.csv:\n", df.head())

# HANDLE MISSING VALUES
# Fill missing dates with mode
df["application_date"] = pd.to_datetime(df["application_date"])
df["decision_date"] = pd.to_datetime(df["decision_date"])
df["application_date"] = df["application_date"].fillna(df["application_date"].mode()[0])
df["decision_date"] = df["decision_date"].fillna(df["decision_date"].mode()[0])

# Fill missing categorical values with 'Unknown'
df["country"] = df["country"].fillna("Unknown")
df["visa_type"] = df["visa_type"].fillna("Unknown")

# CALCULATE PROCESSING DAYS
df["processing_days"] = (df["decision_date"] - df["application_date"]).dt.days
# Any negative processing days → set as NaN
df.loc[df["processing_days"] < 0, "processing_days"] = np.nan


# ADD PROCESSING OFFICE (based on country)
office_map = {
    "India": "New Delhi",
    "United States": "Washington DC",
    "United Kingdom": "London",
    "Canada": "Ottawa",
    "Australia": "Canberra",
    "Germany": "Berlin",
    "France": "Paris",
    "Japan": "Tokyo",
    "Brazil": "Brasilia",
    "Unknown": "Unknown"
}
df["processing_office"] = df["country"].map(office_map)


# FEATURE ENGINEERING

# Application month
df["application_month"] = df["application_date"].dt.month

# Season: Peak vs Off-Peak
df["season"] = df["application_month"].apply(lambda x: "Peak" if x in [1,2,12] else "Off-Peak")

# Country average processing days
country_avg = df.groupby("country")["processing_days"].mean()
df["country_avg"] = df["country"].map(country_avg)
# Fill NaN in country_avg with overall mean
df["country_avg"] = df["country_avg"].fillna(df["processing_days"].mean())

# Visa type average processing days
visa_avg = df.groupby("visa_type")["processing_days"].mean()
df["visa_avg"] = df["visa_type"].map(visa_avg)
# Fill NaN in visa_avg with overall mean
df["visa_avg"] = df["visa_avg"].fillna(df["processing_days"].mean())

print("\nDataFrame after feature engineering:\n", df)

# ENCODING CATEGORICAL FEATURES
df_encoded = pd.get_dummies(df, columns=["country", "visa_type", "season", "processing_office"])
print("\nEncoded DataFrame ready for ML:\n", df_encoded)


# MACHINE LEARNING
# Drop rows with missing processing_days (target variable)
df_ml = df_encoded.dropna(subset=["processing_days"])

X = df_ml.drop(columns=["processing_days", "application_date", "decision_date"])
y = df_ml["processing_days"]

# Fill any remaining NaN values in features with 0 (shouldn't happen, but safety check)
X = X.fillna(0)

model = LinearRegression()
model.fit(X, y)


# PREDICTION SAMPLE
sample_input = pd.DataFrame(np.zeros((1, len(X.columns))), columns=X.columns)
# Example: India + Student + Peak + New Delhi office
sample_input.loc[0, "country_India"] = 1
sample_input.loc[0, "visa_type_Student"] = 1
sample_input.loc[0, "season_Peak"] = 1
sample_input.loc[0, "processing_office_New Delhi"] = 1

predicted_days = model.predict(sample_input)
print("\nPredicted Processing Time (India + Student + Peak + New Delhi):", round(predicted_days[0], 2), "days")

# VISUALIZATIONS
# Filter df to exclude NaN processing_days for visualizations
df_clean = df.dropna(subset=["processing_days"])

sns.histplot(df_clean["processing_days"], kde=True)
plt.title("Distribution of Visa Processing Days")
plt.xlabel("Processing Days")
plt.ylabel("Count")
plt.show()

sns.boxplot(x=df_clean["processing_days"])
plt.title("Boxplot of Processing Days")
plt.show()

sns.scatterplot(x="application_month", y="processing_days", data=df_clean)
plt.title("Processing Days vs Application Month")
plt.show()

corr_matrix = df_clean[["processing_days", "application_month"]].corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
