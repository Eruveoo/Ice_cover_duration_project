import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data_ip = pd.read_csv("data_ip.csv")
lake_data = pd.read_csv("ltbl_ice.csv")

merged_data = data_ip.merge(lake_data, on="station_id", how="left")

min_lat = merged_data["lat_wgs84"].min()
max_lat = merged_data["lat_wgs84"].max()

bin_width = 5
latitude_bins = np.arange(min_lat, max_lat + bin_width, bin_width)
merged_data["latitude_bin"] = pd.cut(merged_data["lat_wgs84"], bins=latitude_bins, labels=latitude_bins[:-1])

records_per_latitude = merged_data.groupby(["latitude_bin", "year"]).size().reset_index(name="record_count")

pivot_data = records_per_latitude.pivot(index="year", columns="latitude_bin", values="record_count")

plt.figure(figsize=(12, 6))
for latitude in pivot_data.columns:
    plt.plot(pivot_data.index, pivot_data[latitude], label=f"{int(latitude)}° to {int(latitude+bin_width)}°", marker='o', linestyle='-')

plt.xlabel("Year")
plt.ylabel("Number of Records")
plt.title("Number of Ice Cover Records per Year by Latitude")
plt.legend(title="Latitude Range (°)", loc="upper right", fontsize=8, ncol=2)
plt.grid(True)

plt.show()
