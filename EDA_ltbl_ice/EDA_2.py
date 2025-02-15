import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

data_ip = pd.read_csv("data_ip.csv")
lake_data = pd.read_csv("ltbl_ice.csv")

merged_data = data_ip.merge(lake_data, on="station_id", how="left")

lake_counts = merged_data.groupby(["lat_wgs84", "lon_wgs84"]).size().reset_index(name="record_count")

gdf = gpd.GeoDataFrame(lake_counts, geometry=gpd.points_from_xy(lake_counts["lon_wgs84"], lake_counts["lat_wgs84"]))

world_map_path = "110m_cultural/ne_110m_admin_0_countries.shp"
world = gpd.read_file(world_map_path)

scandinavia = gdf[(gdf["lat_wgs84"] > 55) & (gdf["lon_wgs84"] > 0) & (gdf["lon_wgs84"] < 40)]
north_america = gdf[(gdf["lat_wgs84"] > 30) & (gdf["lon_wgs84"] < -50)]

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

world.plot(ax=axes[0], color="lightgrey", edgecolor="black")
scandinavia_scatter = axes[0].scatter(
    scandinavia.geometry.x, scandinavia.geometry.y,
    s=scandinavia["record_count"] * 0.5,
    c=scandinavia["record_count"], cmap="coolwarm", alpha=0.75, edgecolors="black"
)
axes[0].set_xlim(-10, 40)
axes[0].set_ylim(55, 75)
axes[0].set_title("Ice Cover Records in Fennoscandia")
axes[0].set_xlabel("Longitude")
axes[0].set_ylabel("Latitude")

cbar1 = fig.colorbar(scandinavia_scatter, ax=axes[0], orientation="vertical", fraction=0.04, pad=0.05)
cbar1.set_label("Number of Records")

world.plot(ax=axes[1], color="lightgrey", edgecolor="black")
north_america_scatter = axes[1].scatter(
    north_america.geometry.x, north_america.geometry.y,
    s=north_america["record_count"] * 0.5,
    c=north_america["record_count"], cmap="coolwarm", alpha=0.75, edgecolors="black"
)
axes[1].set_xlim(-160, -50)
axes[1].set_ylim(30, 80)
axes[1].set_title("Ice Cover Records in North America")
axes[1].set_xlabel("Longitude")
axes[1].set_ylabel("Latitude")

cbar2 = fig.colorbar(north_america_scatter, ax=axes[1], orientation="vertical", fraction=0.04, pad=0.05)
cbar2.set_label("Number of Records")

plt.tight_layout()
plt.show()
