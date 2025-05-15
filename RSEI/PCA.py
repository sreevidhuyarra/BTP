import rasterio
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# === Step 1: Load raster files ===
file_paths = {
    'NDVI': 'C:\\Users\\DELL\\Downloads\\NDVI_July_2022_Krishna.tif',
    'NDWI': 'C:\\Users\\DELL\\OneDrive\\Desktop\\Documents\\BTP\\Results_July_2022\\ndwi_2022_07.tif',
    'EVI': 'C:\\Users\\DELL\\Downloads\\EVI_July_2022_Krishna.tif',
    'TCI': 'C:\\Users\\DELL\\OneDrive\\Desktop\\Documents\\BTP-TCI\\Results_July_2022\\tci_2022_07.tif',
    'MNDWI': 'C:\\Users\\DELL\\Downloads\\MNDWI_July_2022_Krishna.tif'
}

data_arrays = []
profile = None

# Read all rasters and store as arrays
for i, (index_name, file_path) in enumerate(file_paths.items()):
    with rasterio.open(file_path) as src:
        if profile is None:
            profile = src.profile
        array = src.read(1).astype('float32')
        array[array == src.nodata] = np.nan
        data_arrays.append(array)

# Stack the arrays: shape = (rows, cols, 5)
stacked_data = np.stack(data_arrays, axis=-1)
rows, cols, bands = stacked_data.shape

# === Step 2: Flatten and remove NaNs ===
flat_data = stacked_data.reshape(-1, bands)
valid_mask = ~np.isnan(flat_data).any(axis=1)
valid_data = flat_data[valid_mask]

# === Step 3: Normalize the valid data ===
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(valid_data)

# === Step 4: PCA ===
pca = PCA(n_components=1)
principal_component = pca.fit_transform(normalized_data).flatten()

# === Step 5: Normalize the first principal component to 0-1 (RSEI) ===
rsei = (principal_component - principal_component.min()) / (principal_component.max() - principal_component.min())

# === Step 6: Reshape to original spatial structure ===
rsei_image = np.full(flat_data.shape[0], np.nan)
rsei_image[valid_mask] = rsei
rsei_image = rsei_image.reshape(rows, cols)

# === Step 7: Save the RSEI result ===
profile.update(dtype='float32', count=1, nodata=np.nan)

with rasterio.open('Original_RSEI_July2022.tif', 'w', **profile) as dst:
    dst.write(rsei_image.astype('float32'), 1)

# Optional: Plot RSEI
plt.imshow(rsei_image, cmap='YlGn', vmin=0, vmax=1)
plt.colorbar(label='RSEI')
plt.title('Original RSEI - July 2022')
plt.savefig("original_RSEI_July_2022.png", dpi=300)
plt.show()
