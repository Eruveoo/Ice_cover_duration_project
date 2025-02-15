import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, TimeSeriesSplit

data_ip = pd.read_csv("data_ip.csv")
lake_data = pd.read_csv("ltbl_ice.csv")

merged_data = data_ip.merge(lake_data, on="station_id", how="left")

merged_data['ice_on'] = pd.to_datetime(merged_data['ice_on'], errors='coerce')
merged_data['ice_off'] = pd.to_datetime(merged_data['ice_off'], errors='coerce')

merged_data = merged_data.dropna(subset=['ice_on', 'ice_off'])

merged_data['ice_duration'] = (merged_data['ice_off'] - merged_data['ice_on']).dt.days

merged_data = merged_data[merged_data["ice_duration"] >= 0]

features = ["year"]
X = merged_data[features]
y = merged_data["ice_duration"]

scaler = MinMaxScaler(feature_range=(-1, 1))
X_scaled = scaler.fit_transform(X)

cv = KFold(n_splits=10, shuffle=True, random_state=42)
#cv = TimeSeriesSplit(n_splits=10)

year_counts = X["year"].value_counts()
weights = 1 / X["year"].map(year_counts)

weights /= weights.sum()

best_models = {}
results = []

for degree in range(11):
    if degree == 0:
        X_poly = np.ones((X_scaled.shape[0], 1))
        feature_names = ["Intercept"]
    else:
        poly_transformer = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly_transformer.fit_transform(X_scaled)
        feature_names = ["Intercept"] + list(poly_transformer.get_feature_names_out(["scaled_year"]))

    mse_scores = []

    for train_index, test_index in cv.split(X_poly):
        X_train, X_test = X_poly[train_index], X_poly[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        weights_train = weights.iloc[train_index]
        weights_test = weights.iloc[test_index]

        ridge = Ridge(alpha=0)
        ridge.fit(X_train, y_train, sample_weight=weights_train)

        y_pred = ridge.predict(X_test)
        mse = np.average((y_pred - y_test) ** 2, weights=weights_test)
        mse_scores.append(mse)
    cv_loss = np.mean(mse_scores)

    best_models[degree] = ridge

    coefs = ridge.coef_
    intercept = ridge.intercept_
    all_coefficients = [intercept] + list(coefs)

    results.append([degree, cv_loss] + all_coefficients)

coef_columns = ["Degree", "CV Loss"] + feature_names
results_df = pd.DataFrame(results, columns=coef_columns)

print(results_df)

year_range = np.linspace(X.min(), X.max(), 300)
year_range_scaled = scaler.transform(year_range)

plt.figure(figsize=(12, 7))

for degree in range(11):
    if degree == 0:
        X_poly = np.ones((year_range_scaled.shape[0], 1))
    else:
        poly_transformer = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly_transformer.fit_transform(year_range_scaled)

    best_model = best_models[degree]
    ice_duration_pred = best_model.predict(X_poly)

    plt.plot(year_range, ice_duration_pred, linewidth=2, label=f"Degree {degree}")

plt.scatter(X, y, color='black', alpha=0.5, label="Observed Data")

plt.xlabel("Year")
plt.ylabel("Ice Duration (days)")
plt.title("Fitted Polynomial Models of Different Degrees with Adjusted Loss")
plt.legend()
plt.grid(True)
plt.show()

fig, ax = plt.subplots(figsize=(12, len(results_df) * 0.5))

ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
ax.set_frame_on(False)

table = ax.table(cellText=results_df.values, colLabels=results_df.columns, cellLoc='center', loc='center')

table.auto_set_font_size(False)
table.set_fontsize(10)
table.auto_set_column_width([i for i in range(len(results_df.columns))])

plt.show()

plt.figure(figsize=(12, 7))

for degree in range(11):
    if degree == 0:
        X_poly = np.ones((year_range_scaled.shape[0], 1))
    else:
        poly_transformer = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly_transformer.fit_transform(year_range_scaled)

    best_model = best_models[degree]
    ice_duration_pred = best_model.predict(X_poly)

    plt.plot(year_range, ice_duration_pred, linewidth=2, label=f"Degree {degree}")

plt.xlabel("Year")
plt.ylabel("Ice Duration (days)")
plt.title("Fitted Polynomial Models of Different Degrees")
plt.legend()
plt.grid(True)
plt.show()
