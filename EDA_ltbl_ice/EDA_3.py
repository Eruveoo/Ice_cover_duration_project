import pandas as pd
import matplotlib.pyplot as plt

data_ip = pd.read_csv("data_ip.csv")
lake_data = pd.read_csv("ltbl_ice.csv")

merged_data = data_ip.merge(lake_data, on="station_id", how="left")

records_per_year = merged_data.groupby(["country", "year"]).size().reset_index(name="record_count")

pivot_data = records_per_year.pivot(index="year", columns="country", values="record_count")

plt.figure(figsize=(12, 6))
for country in pivot_data.columns:
    plt.plot(pivot_data.index, pivot_data[country], label=country, marker='o', linestyle='-')

plt.xlabel("Year")
plt.ylabel("Number of Records")
plt.title("Number of Ice Cover Records per Year by Country")
plt.legend(title="Country")
plt.grid(True)

plt.show()
