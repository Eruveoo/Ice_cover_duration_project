import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data_ip.csv")

df['ice_on'] = pd.to_datetime(df['ice_on'], errors='coerce')
df['ice_off'] = pd.to_datetime(df['ice_off'], errors='coerce')

df = df.dropna(subset=['ice_on', 'ice_off'])

df['year'] = df['ice_on'].dt.year

first_year = df['year'].min()
last_year = df['year'].max()
total_years = last_year - first_year

station_coverage = df.groupby('station_id')['year'].nunique().reset_index()
station_coverage.columns = ['station_id', 'years_measured']

station_coverage['percentage_measured'] = (station_coverage['years_measured'] / total_years) * 100

plt.figure(figsize=(10, 6))
plt.hist(station_coverage['percentage_measured'], bins=20, color='b', alpha=0.7, edgecolor='black')
plt.xlabel("Percentage of Total Available Years Measured", fontsize=18, labelpad=10)
plt.ylabel("Number of Stations", fontsize=18, labelpad=10)
plt.title("Histogram of Station Data Completeness", fontsize=22, pad=15)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()
