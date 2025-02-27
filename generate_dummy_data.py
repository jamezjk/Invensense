import pandas as pd
import numpy as np
import datetime

# Generate 60 days of dummy demand data
dates = pd.date_range(start="2023-06-01", periods=60, freq="D")
demand = np.random.randint(50, 200, size=(60,))  # Random demand values between 50-200

# Create DataFrame
df = pd.DataFrame({"Date": dates, "Demand": demand})
df.to_csv("data.csv", index=False)

print("✅ Dummy data.csv file created successfully!")
