import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, KFold

data_ip = pd.read_csv("data_ip.csv")
lake_data = pd.read_csv("ltbl_ice.csv")

merged_data = data_ip.merge(lake_data, on="station_id", how="left")

merged_data['ice_on'] = pd.to_datetime(merged_data['ice_on'], errors='coerce')
merged_data['ice_off'] = pd.to_datetime(merged_data['ice_off'], errors='coerce')

merged_data = merged_data.dropna(subset=['ice_on', 'ice_off'])

merged_data['ice_duration'] = (merged_data['ice_off'] - merged_data['ice_on']).dt.days

merged_data = merged_data[merged_data["ice_duration"] < 1000]

features = ["year", "lat_wgs84", "altitude_m", "country"]
X = merged_data[features]
y = merged_data["ice_duration"]

numerical_features = ["year", "lat_wgs84", "altitude_m"]

categorical_features = ["country"]

year_scaler = MinMaxScaler(feature_range=(-1, 1))
X["year"] = year_scaler.fit_transform(X[["year"]])

other_numerical_features = [col for col in numerical_features if col != "year"]
scaler = StandardScaler()
X[other_numerical_features] = scaler.fit_transform(X[other_numerical_features])

encoder = OneHotEncoder(sparse_output=False)
X_encoded = encoder.fit_transform(X[categorical_features])
encoded_feature_names = encoder.get_feature_names_out(categorical_features)

X_encoded_df = pd.DataFrame(X_encoded, columns=encoded_feature_names, index=X.index)

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X[numerical_features])

poly_feature_names = poly.get_feature_names_out(numerical_features)
X_poly_df = pd.DataFrame(X_poly, columns=poly_feature_names, index=X.index)

X_interactions = pd.DataFrame(index=X.index)

for cat_col in X_encoded_df.columns:
    for num_col in numerical_features:
        X_interactions[f"{cat_col} × {num_col}"] = X_encoded_df[cat_col] * X[num_col]

X_final = pd.concat([X_poly_df, X_encoded_df, X_interactions], axis=1)

cv = KFold(n_splits=10, shuffle=True, random_state=42)

alpha_grid = np.logspace(-5, 1, 50)

pipeline = Pipeline([
    ("ridge", Ridge())
])

grid_search = GridSearchCV(pipeline, {"ridge__alpha": alpha_grid}, scoring="neg_mean_squared_error", cv=cv, return_train_score=True)
grid_search.fit(X_final, y)

# Extract best model from grid search
best_ridge_model = grid_search.best_estimator_
ridge_model = best_ridge_model.named_steps["ridge"]

# Extract feature names
feature_names = X_final.columns

# Extract raw coefficients
coefs = ridge_model.coef_

# Extract the intercept
intercept = ridge_model.intercept_

# Create a DataFrame for coefficients
coef_df = pd.DataFrame({"Feature": feature_names, "Coefficient": coefs})
coef_df = coef_df.sort_values(by="Coefficient", ascending=False)

# Print the intercept
print("\nIntercept of the Model:", round(intercept, 4))

print("\nFull Coefficient Table:")
print(coef_df.to_string(index=False))  # Print all coefficients nicely formatted

cv_results = grid_search.cv_results_
alphas = cv_results["param_ridge__alpha"].data.astype(float)
mean_scores = -cv_results["mean_test_score"]

plt.figure(figsize=(10, 6))
plt.plot(alphas, mean_scores, marker="o", linestyle="-", label="Cross-Validation Loss")
plt.xscale("log")
plt.xlabel("Alpha (Regularization Strength)")
plt.ylabel("Cross-Validation MSE")
plt.title("Tuning of Alpha Parameter in Ridge Regression")
plt.legend()
plt.grid(True)

plt.show()

time_effects = coef_df[coef_df["Feature"].str.match(r"^(year|\byear\^2\b|.* × year)$")]
country_effects = coef_df[coef_df["Feature"].str.match(r"^country_[A-Z]{2}$")]
country_interactions = coef_df[coef_df["Feature"].str.contains("country_") & coef_df["Feature"].str.contains(" × ") & ~coef_df["Feature"].str.contains("year")]
other_effects = coef_df[~coef_df["Feature"].str.contains("year|country_")]

time_effects = time_effects.sort_values(by="Coefficient", ascending=False)
country_effects = country_effects.sort_values(by="Coefficient", ascending=False)
country_interactions = country_interactions.sort_values(by="Coefficient", ascending=False)
other_effects = other_effects.sort_values(by="Coefficient", ascending=False)

print(f"\nIntercept: {round(intercept, 4)}")

print("\n==== Time Effects (Year-Based) ====")
print(time_effects.to_string(index=False))

print("\n==== Country-Specific Effects ====")
for country in country_effects["Feature"]:
    print(f"\nCountry: {country.replace('country_', '')}")
    print(country_effects[country_effects["Feature"] == country].to_string(index=False))
    country_interactions_sub = country_interactions[country_interactions["Feature"].str.contains(country)]
    if not country_interactions_sub.empty:
        print(country_interactions_sub.to_string(index=False))

print("\n==== Other Effects (Geospatial, Environmental, and Squared Terms) ====")
print(other_effects.to_string(index=False))
