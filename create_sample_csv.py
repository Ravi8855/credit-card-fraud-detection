import pandas as pd

# Load your full dataset
df = pd.read_csv("creditcard.csv")

# Create 1000-row sample (200 fraud, 800 non-fraud)
fraud = df[df["Class"] == 1].sample(n=200, random_state=42)
non_fraud = df[df["Class"] == 0].sample(n=800, random_state=42)

# Combine and shuffle
sample_df = pd.concat([fraud, non_fraud]).sample(frac=1, random_state=42)

# Save sample dataset
sample_df.to_csv("sample_creditcard.csv", index=False)

print("sample_creditcard.csv created successfully!")
