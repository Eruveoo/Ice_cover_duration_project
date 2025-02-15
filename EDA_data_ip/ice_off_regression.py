import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data_ip.csv")

df['ice_off'] = pd.to_datetime(df['ice_off'], errors='coerce')

df = df.dropna(subset=['ice_off'])

df['ice_off_day'] = df['ice_off'].dt.dayofyear
df['ice_off_year'] = df['ice_off'].dt.year

df.loc[df['ice_off_year'] < df['year'], 'ice_off_day'] -= 365

yearly_avg = df.groupby('year')['ice_off_day'].mean().reset_index()

x = yearly_avg['year'].values
y = yearly_avg['ice_off_day'].values

coefs = np.polyfit(x, y, 2)
poly_fit = np.poly1d(coefs)

x_fit = np.linspace(x.min(), x.max(), 200)
y_fit = poly_fit(x_fit)

n_bootstrap = 1000
bootstrap_curves = np.zeros((n_bootstrap, len(x_fit)))

for i in range(n_bootstrap):
    sample_idx = np.random.choice(len(x), len(x), replace=True)
    sample_x, sample_y = x[sample_idx], y[sample_idx]
    sample_coefs = np.polyfit(sample_x, sample_y, 2)
    sample_poly = np.poly1d(sample_coefs)
    bootstrap_curves[i, :] = sample_poly(x_fit)

y_lower = np.percentile(bootstrap_curves, 5, axis=0)
y_upper = np.percentile(bootstrap_curves, 95, axis=0)

plt.figure(figsize=(12, 6))
plt.plot(x, y, marker='o', linestyle='-', color='b', label="Yearly Avg Ice Off Day")
plt.plot(x_fit, y_fit, linestyle='--', color='r', label="2nd Order Polynomial Fit")
plt.fill_between(x_fit, y_lower, y_upper, color='r', alpha=0.2, label="Confidence Interval")
plt.xlabel("Year", fontsize=18, labelpad=10)
plt.ylabel("Average Ice Off Day (Day of Year)", fontsize=18, labelpad=10)
plt.title("Trend of Ice Off Day Over Time with Confidence Interval", fontsize=22, pad=15)
plt.legend()
plt.grid(True)
plt.show()
