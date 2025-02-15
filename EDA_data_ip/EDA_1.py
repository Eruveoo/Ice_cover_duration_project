import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data_ip.csv")

df['ice_on'] = pd.to_datetime(df['ice_on'], errors='coerce')
df['ice_off'] = pd.to_datetime(df['ice_off'], errors='coerce')

total_records = len(df)
both_present = df.dropna(subset=['ice_on', 'ice_off']).shape[0]
missing_ice_on = df['ice_on'].isna().sum()
missing_ice_off = df['ice_off'].isna().sum()
both_missing = df[df['ice_on'].isna() & df['ice_off'].isna()].shape[0]

print(f"Total records: {total_records}")
print(f"Records with both ice_on and ice_off: {both_present}")
print(f"Records with missing ice_on: {missing_ice_on} ({(missing_ice_on / total_records) * 100:.2f}%)")
print(f"Records with missing ice_off: {missing_ice_off} ({(missing_ice_off / total_records) * 100:.2f}%)")
print(f"Records with both ice_on and ice_off missing: {both_missing} ({(both_missing / total_records) * 100:.2f}%)")

if both_missing > 0:
    print("Warning: There are records where both ice_on and ice_off are missing!")

# Convert ice_on and ice_off to datetime format
df['ice_on'] = pd.to_datetime(df['ice_on'], errors='coerce')
df['ice_off'] = pd.to_datetime(df['ice_off'], errors='coerce')

df['year_on'] = df['ice_on'].dt.year
df['year_off'] = df['ice_off'].dt.year

station_ranges = df.groupby("station_id")[['year_on', 'year_off']].agg(['min', 'max']).reset_index()
station_ranges.columns = ['station_id', 'start_on', 'end_on', 'start_off', 'end_off']

station_ranges['start_year'] = station_ranges[['start_on', 'start_off']].min(axis=1)
station_ranges['end_year'] = station_ranges[['end_on', 'end_off']].max(axis=1)

station_ranges = station_ranges.sort_values(by="start_year").reset_index(drop=True)

plt.figure(figsize=(12, 8))
for i, row in station_ranges.iterrows():
    plt.plot([row['start_year'], row['end_year']], [i, i], marker='o', linestyle='-')

plt.xlabel("Year", fontsize=18, labelpad=10)
plt.ylabel("Stations (Ordered by First Year of Data)", fontsize=18, labelpad=10)
plt.title("Time Coverage of Lake Ice Data by Station", fontsize=22, pad=15)

plt.grid(True)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.grid(True)
plt.show()

df = pd.read_csv("data_ip.csv")

df['lake_id'] = df['station_id'].apply(lambda x: x.split('_')[0])

lake_station_counts = df.groupby('lake_id')['station_id'].nunique()

station_distribution = lake_station_counts.value_counts().sort_index()

print(station_distribution)

print("\nLakes with multiple stations:")
for num_stations in sorted(station_distribution.index):
    if num_stations >= 2:
        lakes = lake_station_counts[lake_station_counts == num_stations].index.tolist()
        print(f"{num_stations} stations: {', '.join(lakes)}")
