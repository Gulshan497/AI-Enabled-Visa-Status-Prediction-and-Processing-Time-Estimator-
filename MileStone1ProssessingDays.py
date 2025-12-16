import numpy as np
import pandas as pd
# Step 1: Create data
data = {
    "Application_Date": [
        "2024-01-01", "2024-02-15", "2024-03-10",
        "2024-04-05", "2024-05-12", "2024-06-01",
        "2024-06-18", "2024-07-02", "2024-07-25",
        "2024-08-10"
    ],
    "Decision_Date": [
        "2024-02-01", "2024-03-20", "2024-04-05",
        "2024-05-01", "2024-06-20", "2024-06-25",
        "2024-07-28", "2024-08-05", "2024-09-15",
        "2024-10-01"
    ],
    "Country": [
        "India", "United States", "United Kingdom",
        "Canada", "Australia", "Germany",
        "India", "France", "Japan", "Brazil"
    ],
    "Visa_Type": [
        "Student", "Tourist", "Work",
        "Tourist", "Student", "Work",
        "Work", "Tourist", "Student", "Work"
    ]
}

df = pd.DataFrame(data)
# Step 2: Ensure required columns exist
required_columns = ["Application_Date", "Decision_Date", "Country", "Visa_Type"]

for col in required_columns:
    if col not in df.columns:
        df[col] = np.nan

# Step 3: Handle missing categorical data
df["Country"] = df["Country"].fillna("Unknown")
df["Visa_Type"] = df["Visa_Type"].fillna("Unknown")

# Step 4: Handle dates safely

df["Application_Date"] = pd.to_datetime(df["Application_Date"], errors="coerce")
df["Decision_Date"] = pd.to_datetime(df["Decision_Date"], errors="coerce")

# Step 5: Calculate processing days safely

df["Processing_Days"] = (
    df["Decision_Date"] - df["Application_Date"]
).dt.days

df.loc[df["Processing_Days"] < 0, "Processing_Days"] = np.nan


# Step 6: Map processing office

office_map = {
    "India": "New Delhi",
    "United States": "Washington DC",
    "United Kingdom": "London",
    "Canada": "Ottawa",
    "Australia": "Canberra",
    "Germany": "Berlin",
    "France": "Paris",
    "Japan": "Tokyo",
    "Brazil": "Brasilia"
}

df["Processing_Office"] = df["Country"].map(office_map).fillna("Unknown")

# Step 7: Display

pd.set_option("display.max_columns", None)
print(df)
