import pandas as pd
import matplotlib.pyplot as plt

data_ip = pd.read_csv("data_ip.csv")
lake_data = pd.read_csv("ltbl_ice.csv")

merged_data = data_ip.merge(lake_data, on="station_id", how="left")

mean_latitude_per_year = merged_data.groupby("year")["lat_wgs84"].mean().reset_index()

plt.figure(figsize=(12, 6))
plt.plot(mean_latitude_per_year["year"], mean_latitude_per_year["lat_wgs84"], marker='o', linestyle='-', color="royalblue")

plt.xlabel("Year")
plt.ylabel("Average Latitude (°)")
plt.title("Change in Average Latitude of Ice Cover Records Over Time")
plt.grid(True)

plt.show()
