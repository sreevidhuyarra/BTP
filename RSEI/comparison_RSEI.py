import numpy as np
import rasterio
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Load both rasters
predicted_path = r"C:\Users\DELL\OneDrive\Desktop\Documents\BTP\PCA\predicted_RSEI_July2022.tif"
actual_path = r"C:\Users\DELL\OneDrive\Desktop\Documents\BTP\PCA\Original_RSEI_July2022.tif"  # Adjust to actual path

with rasterio.open(predicted_path) as src:
    predicted = src.read(1)
    profile = src.profile





with rasterio.open(actual_path) as src:
    actual = src.read(1)

# 2. Visual comparison - Side by side
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# Predicted NDWI
im1 = ax1.imshow(predicted, cmap='YlGn', vmin=-1, vmax=1)
ax1.set_title('Predicted RSEI - July 2022')
ax1.axis('off')
plt.colorbar(im1, ax=ax1, label='RSEI')

# Actual NDWI
im2 = ax2.imshow(actual, cmap='YlGn', vmin=-1, vmax=1)
ax2.set_title('Actual RSEI - July 2022')
ax2.axis('off')
plt.colorbar(im2, ax=ax2, label='RSEI')

# Compute squared error per pixel, ignoring NaNs
error_squared = np.where(~np.isnan(predicted) & ~np.isnan(actual), (actual - predicted) ** 2, np.nan)
rmse_map = np.sqrt(error_squared)  # This is effectively abs(error) for a single timestep

# Plot RMSE map
im3 = ax3.imshow(rmse_map, cmap='Reds', vmin=0, vmax=0.5)
ax3.set_title('Per-Pixel RMSE (|Actual - Predicted|)')
ax3.axis('off')
plt.colorbar(im3, ax=ax3, label='RMSE')

plt.tight_layout()
plt.savefig("tci_01_2025.png", dpi=300)
plt.show()

# # 3. Statistical comparison
# # Flatten arrays and remove NaN values
# pred_flat = predicted.flatten()
# actual_flat = actual.flatten()
# valid_pixels = ~(np.isnan(pred_flat) | np.isnan(actual_flat))
# pred_valid = pred_flat[valid_pixels]
# actual_valid = actual_flat[valid_pixels]

# # Calculate error metrics
# rmse = np.sqrt(mean_squared_error(actual_valid, pred_valid))
# mae = mean_absolute_error(actual_valid, pred_valid)
# r2 = r2_score(actual_valid, pred_valid)

# print(f"Root Mean Square Error (RMSE): {rmse:.4f}")
# print(f"Mean Absolute Error (MAE): {mae:.4f}")
# print(f"R² Score: {r2:.4f}")

# # 4. Create a histogram of errors
# plt.figure(figsize=(10, 6))
# plt.hist(difference[~np.isnan(difference)].flatten(), bins=50, alpha=0.75)
# plt.title('Histogram of Prediction Errors (July 2022)')
# plt.xlabel('Error (Actual - Predicted)')
# plt.ylabel('Frequency')
# plt.grid(alpha=0.3)
# plt.savefig("predicted_RSEI_July_2022_histogram_of_errors.png", dpi=300)
# plt.show()

# # 4. Histogram of per-pixel squared errors (MSE values)
# squared_error = (actual - predicted) ** 2
# valid_squared_error = squared_error[~np.isnan(squared_error)]

# plt.figure(figsize=(10, 6))
# plt.hist(valid_squared_error.flatten(), bins=50, alpha=0.75, color='orange')
# plt.title('Histogram of Per-Pixel Squared Errors (MSE) - July 2022')
# plt.xlabel('Squared Error')
# plt.ylabel('Frequency')
# plt.grid(alpha=0.3)
# plt.tight_layout()
# plt.savefig("predicted_RSEI_July_2022_histogram_MSE.png", dpi=300)
# plt.show()

# # 5. Scatter plot (actual vs predicted)
# plt.figure(figsize=(10, 10))
# plt.scatter(actual_valid, pred_valid, alpha=0.1, s=1)
# plt.plot([-1, 1], [-1, 1], 'r--')  # Perfect prediction line
# plt.xlabel('Actual TCI')
# plt.ylabel('Predicted TCI')
# plt.title('Predicted vs Actual TCI Values (July 2022)')
# plt.axis('equal')
# plt.grid(alpha=0.3)
# plt.savefig("RSEI_July_2022_actual_vs_predicted.png", dpi=300)
# plt.show()