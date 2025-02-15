import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

lake_data = pd.read_csv("ltbl_ice.csv")

country_counts = lake_data['country'].value_counts()

print(country_counts)

for column in lake_data.columns:
    missing_count = lake_data[column].isnull().sum()
    total_count = len(lake_data[column])
    missing_percentage = (missing_count / total_count) * 100

    print(f"Column: {column}")
    print(f"  Missing Values: {missing_count}")
    print(f"  Total Records: {total_count}")
    print(f"  Percentage Missing: {missing_percentage:.2f}%")
    print("-" * 40)

lake_data = lake_data.drop(columns=["lake_id", "cent_lat_wgs84", "cent_lon_wgs84"])

lake_data["country"] = lake_data["country"].astype("category")

lake_data["area_ha_log"] = np.log1p(lake_data["area_ha"])
lake_data["depth_max_m_log"] = np.log1p(lake_data["depth_max_m"])

num_cols = lake_data.select_dtypes(include=['number'])

g = sns.pairplot(lake_data, diag_kind="hist", hue="country", palette="Set2", plot_kws={'alpha': 0.6})

for t, label in zip(g._legend.texts, lake_data["country"].cat.categories):
    t.set_text(label)

plt.show()