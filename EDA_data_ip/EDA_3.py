import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data_ip.csv")

df['ice_on'] = pd.to_datetime(df['ice_on'], errors='coerce')
df['ice_off'] = pd.to_datetime(df['ice_off'], errors='coerce')

df['year'] = df['ice_on'].dt.year

yearly_counts = df['year'].value_counts().sort_index()

plt.figure(figsize=(12, 6))
plt.bar(yearly_counts.index, yearly_counts.values, color='b', alpha=0.7)
plt.xlabel("Year", fontsize=18, labelpad=10)
plt.ylabel("Number of Measurements", fontsize=18, labelpad=10)
plt.title("Number of Ice Cover Measurements Per Year", fontsize=22, pad=15)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
