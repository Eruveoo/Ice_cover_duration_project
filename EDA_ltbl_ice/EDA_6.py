import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

data_ip = pd.read_csv("data_ip.csv")
lake_data = pd.read_csv("ltbl_ice.csv")

merged_data = data_ip.merge(lake_data, on="station_id", how="left")

world_map_path = "110m_cultural/ne_110m_admin_0_countries.shp"
world = gpd.read_file(world_map_path)

years = sorted(merged_data["year"].unique())

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

def update(year):
    axes[0].cla()
    axes[1].cla()

    yearly_data = merged_data[merged_data["year"] == year]

    gdf = gpd.GeoDataFrame(yearly_data, geometry=gpd.points_from_xy(yearly_data["lon_wgs84"], yearly_data["lat_wgs84"]))

    scandinavia = gdf[(gdf["lat_wgs84"] > 55) & (gdf["lon_wgs84"] > 0) & (gdf["lon_wgs84"] < 40)]
    north_america = gdf[(gdf["lat_wgs84"] > 30) & (gdf["lon_wgs84"] < -50)]

    world.plot(ax=axes[0], color="lightgrey", edgecolor="black")
    if not scandinavia.empty:
        axes[0].scatter(
            scandinavia.geometry.x, scandinavia.geometry.y,
            s=20, color="red", alpha=0.75, edgecolors="black"
        )
    axes[0].set_xlim(-10, 40)
    axes[0].set_ylim(55, 75)
    axes[0].set_title(f"Ice Cover Records in Fennoscandia - {year}")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")

    world.plot(ax=axes[1], color="lightgrey", edgecolor="black")
    if not north_america.empty:
        axes[1].scatter(
            north_america.geometry.x, north_america.geometry.y,
            s=20, color="blue", alpha=0.75, edgecolors="black"
        )
    axes[1].set_xlim(-160, -50)
    axes[1].set_ylim(30, 80)
    axes[1].set_title(f"Ice Cover Records in North America - {year}")
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")

ani = animation.FuncAnimation(fig, update, frames=years, repeat=True, interval=500)

ani.save("ice_cover_animation.mp4", writer="ffmpeg", fps=5)

plt.show()
