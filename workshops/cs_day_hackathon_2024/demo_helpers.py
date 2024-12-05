import io
import math
import os
import tarfile
import tempfile
import warnings
from io import BytesIO
from pathlib import Path

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rasterio.io as rio
import rasterio.plot
import xarray as xr
import yaml
from geopandas.geodataframe import GeoDataFrame
from matplotlib.colors import Normalize
from pyproj import Transformer

RasterCube = xr.DataArray


# -----------------------------------------------------------------------------
#                               showfig
# -----------------------------------------------------------------------------
def showfig(data, flag=None, figsize=(5, 5), ax=None):
    """
    Display an image with optional flag-based masking and colorbar.

    Parameters:
    - data : array-like - The data to display.
    - flag : int or None - If provided, applies a bitwise mask on the data.
    - figsize : tuple - The figure size (width, height).
    - ax : matplotlib.axes.Axes or None - Axis to plot on; creates a new one if None.

    Returns:
    - ax : matplotlib.axes.Axes - The axis with the displayed image.
    """
    # Determine if a new axis and colorbar are needed
    create_new_ax = ax is None
    if create_new_ax:
        fig, ax = plt.subplots(figsize=figsize)

    # Apply flag if provided, otherwise use data directly
    img = (data & flag) == flag if flag is not None else data
    cax = ax.imshow(img)

    # Add colorbar only if a new axis was created
    if create_new_ax:
        plt.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

    return ax


# -----------------------------------------------------------------------------
#                                 normalize
# -----------------------------------------------------------------------------
# Normalize the data to the range 0-255 for display
# def normalize(array):
#     array_min, array_max = np.nanpercentile(
#         array, (1, 99)
#     )  # Clip values between 1st and 99th percentile
#     array = np.clip(array, array_min, array_max)  # Clip the extreme values
#     return ((array - array_min) / (array_max - array_min) * 255).astype(np.uint8)

def normalize(array):
    array_min, array_max = np.nanpercentile(array, (1, 99))  # Clip values between 1st and 99th percentile
    array = np.clip(array, array_min, array_max)  # Clip the extreme values
    return (array - array_min) / (array_max - array_min)


# -----------------------------------------------------------------------------
#                                 show_single_result
# -----------------------------------------------------------------------------
def show_single_result(
    image_data, colormap="viridis", is_ndvi=False, title=None, norm=True):
    """
    Visualizes each band of a single GeoTIFF result from OpenEO as a separate image.

    Parameters:
    - image_data : bytes - The single image file response from OpenEO.
    - colormap : str - Colormap for non-NDVI images.
    - is_ndvi : bool - If True, applies an NDVI color map.
    - title : str or None - Title for each image; if None, a default title is generated.
    - norm : bool - Whether to normalize image data before display.

    Returns:
    - A 4D numpy array with shape (timestep, band, x, y).
    """
    if not image_data:
        print("No image data available.")
        return

    images = []
    with io.BytesIO(image_data) as filelike:
        with rasterio.open(filelike) as im:
            if im.count == 0:
                print("The dataset is empty.")
                return

            # Determine the number of bands
            num_bands = im.count
            #print("Reading image data with nodata coded as ", im.nodata)

            # Read each band, and store unnormalized data in `images` for return
            for i in range(1, num_bands + 1):
                band = im.read(i, masked=True)  # Read the band as a masked array

                # Print NoData pixel count before using np.where
                nodata_count = band.mask.sum()  # Count NoData pixels
                total_pixels = band.size  # Total number of pixels
                #print(f"Band {i} has {nodata_count} NoData pixels out of {total_pixels}")

                # Convert the masked array to an ndarray, preserving NoData as NaN
                band_data = np.where(
                    band.mask, np.nan, band.data
                )  # Preserve original values
                
                images.append(band_data)  # Append raw data to `images`

                # Display each band with optional normalization for visualization only
                fig, ax = plt.subplots(figsize=(12, 12))
                cmap = "RdYlGn" if is_ndvi else colormap
                display_data = (
                    normalize(band_data) if norm else band_data
                )  # Use normalized data only for display
                rasterio.plot.show(
                    display_data,
                    ax=ax,
                    cmap=cmap,
                    extent=[
                        im.bounds.left,
                        im.bounds.right,
                        im.bounds.bottom,
                        im.bounds.top,
                    ],
                )
                tags = im.tags()
                bandnames = tags.get("band_names", '0').split(',')
                current_title =  title or f'Band "{bandnames[i-1]}". Timestamp: {tags.get("timestamp", "Unknown")}'
                
                ax.set_title(current_title)
                plt.show()

    # Stack `images` to create a 3D array (band, x, y), and add a new axis for time
    images = np.stack(images)  # Shape is (band, x, y)
    images = images[np.newaxis, ...]  # Add time dimension to get (timestep, band, x, y)

    return images  # Shape is (timestep, band, x, y)



# -----------------------------------------------------------------------------
#                           show_zipped_results
# Visualizes each result of multiple zipped Geotiffs from OpenEO in a single plot.
# The first band of each result is visualized as a subplot.
#
# -----------------------------------------------------------------------------
def show_zipped_results(image_data, colormap="viridis", is_ndvi=False, title=None, norm=True,show_band=0):
    """
    Display images from a compressed tar.gz file containing geotiff images.

    Parameters:
    - image_data : bytes - A single image file response from OpenEO.
    - colormap : str - Colormap to use if `is_ndvi` is False.
    - is_ndvi : bool - Set to True if NDVI calculations were done; applies a specialized color map.
    - title : str or None - Title for the plots.

    Returns:
    - A 4D numpy array with shape (timestep, band, x, y).
    """
    images = []
    transforms = []  # To store the geotransform for each time step
    crs = None  # To store the coordinate reference system (CRS)

    with io.BytesIO(image_data) as filelike, tempfile.TemporaryDirectory() as tmpdirname:
        # Open the tar.gz file
        with tarfile.open(fileobj=filelike, mode="r:gz") as tar:
            print("Unzipping data....")
            tar.extractall(tmpdirname)

            # Determine subdirectory with image files if needed
            subdir = (
                os.path.join(tmpdirname, os.listdir(tmpdirname)[0])
                if len(os.listdir(tmpdirname)) == 1
                else tmpdirname
            )

            # Define allowed image file types and collect them
            image_types = [".tif"]
            ifnames = sorted(
                os.path.join(subdir, f)
                for f in os.listdir(subdir)
                if any(f.endswith(ext) for ext in image_types)
            )

            # Process each image file as a time step
            for ifname in ifnames:
                with rasterio.open(ifname) as src:
                    bands = []

                    for i in range(1, src.count + 1):
                        band_data = src.read(i, masked=True)
                        
                        # Counting NoData pixels
                        nodata_count = band_data.mask.sum()
                        total_pixels = band_data.size
                        print(f"Band {i}: {nodata_count} NoData pixels out of {total_pixels}")

                        bands.append(band_data)  # Append raw data to bands list

                    # Stack bands along a new axis (band, x, y) and add to images list
                    bands = np.stack(bands, axis=0)
                    images.append(bands)

                    # Plotting (with geospatial info)
                    fig, ax = plt.subplots(figsize=(6, 6))
                    cmap = "RdYlGn" if is_ndvi else colormap
                    display_data = (normalize(bands[0]) if norm else bands[0])  
                    
                    rasterio.plot.show(display_data, ax=ax, cmap=cmap, transform=src.transform)
                     
                    tags = src.tags()
                    bandnames = tags.get("band_names", '0').split(',')
                    ax.set_title(
                        title or f'Band "{bandnames[show_band]}". Timestamp: {tags.get("timestamp", "Unknown")}'
                    )
                    plt.show()

    # Stack images along a new axis to form (timestep, band, x, y)
    images = np.stack(images, axis=0)

    # Return images along with geospatial metadata
    return images  # Shape will be (timestep, band, x, y)



# -----------------------------------------------------------------------------
#                               show_result
# -----------------------------------------------------------------------------
def show_result(image_data, colormap="viridis", is_ndvi=False, title=None, norm=True,show_band=0):
    try:
        return show_single_result(image_data, colormap, is_ndvi, title, norm) # Shows all bands always
    except Exception as e:
        pass
        #print("Exception when showing single result file: ",type(e),e)
        #print("Trying to unzip and show the result instead")
    return show_zipped_results(image_data, colormap, is_ndvi, title,norm,show_band)

# -----------------------------------------------------------------------------
#                               plot_pixel_stats
# -----------------------------------------------------------------------------
def plot_pixel_stats(time_band_x_y, title = "Pixel Stats"):
    print(type(time_band_x_y))
    # Assuming results_time_reduce is your 3D array (time, band, x, y)
    # Replace `time_idx` and `band_idx` with the specific time and band you want to plot
    time_idx = 0  # Select the time index
    band_idx = 0  # Select the band index
    
    # Flatten the selected time and band to create 1D array
    data = time_band_x_y[time_idx, band_idx, :, :].flatten()
    
    # Create a figure with subplots: one for the histogram and one for the boxplot
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot the histogram
    axs[0].hist(data, bins=50, color='blue', alpha=0.7)
    axs[0].set_title('Histogram of Pixel Values')
    axs[0].set_xlabel('Pixel Intensity')
    axs[0].set_ylabel('Frequency')
    
    # Plot the boxplot
    axs[1].boxplot(data, vert=False)
    axs[1].set_title('Boxplot of Pixel Values')
    axs[1].set_xlabel('Pixel Intensity')
    
    # Display the plots
    plt.tight_layout()
    fig.suptitle(title,fontsize=16, y=1.05)
    plt.show()
    
# -----------------------------------------------------------------------------
#                               get_s3_wqsf_flags
# -----------------------------------------------------------------------------
def get_s3_wqsf_flags():
    """
    You can get these flags from get_collections, but this is a shortcut for
    training purposes.

    """
    wqsf_flags = {}
    here = Path(__file__).parent
    with open(f"{here}/s3_olci_l2wfr.odc-product.yaml", "r") as stream:
        s3_meta = yaml.safe_load(stream)

        for m in s3_meta["measurements"]:
            if "wqsf" in m["name"]:
                bits = m["flags_definition"]["data"]["values"]
                bitmap = {}
                for b in bits.keys():
                    bitmap[bits[b]] = b

                wqsf_flags[m["name"]] = bitmap
    return wqsf_flags


# -----------------------------------------------------------------------------
#                                plot_xr_DataArray
# -----------------------------------------------------------------------------


def plot_xr_dataarray_contact_copy(
    data: xr.DataArray, cols: int = 4, skip_nans=True, figsize=(10, 15), cmap="viridis"
):
    """nc_data - one band with at least one t."""

    x_min, x_max = data.x.min().item(), data.x.max().item()
    y_min, y_max = data.y.min().item(), data.y.max().item()
    extent = [x_min, x_max, y_min, y_max]

    rows = len(data.t) // cols
    if len(data.t) % cols != 0:
        rows += 1

    fig, axs = plt.subplots(rows, cols, figsize=figsize, constrained_layout=True)
    axs = axs.flatten()  # Flatten to make indexing easier in a loop
    total_pixels = data.y.size * data.x.size
    for i, ax in enumerate(axs):
        if i < len(data.t):
            im = data.isel(t=i)
            if im.count() > 0 or not skip_nans:
                ax.imshow(im, extent=extent, cmap=cmap)
                valid_pixels = im.count().item()
                mean_value = im.reduce(np.nanmean).values.item()
                fraction = round(100 * (valid_pixels / total_pixels), 2)
                ax.set_title(
                    f"{str(data.t.values[i]).split('T')[0]} Valid={fraction}% Mean={round(mean_value,2)}"
                )
                # print(im.count())
                ax.axis("on")  # Or 'off' depending on your preference
            else:
                ax.set_title(f"{str(data.t.values[i]).split('T')[0]} All NaNs")

        else:
            ax.axis("off")  # Hide unused subplots

    plt.show()


# -------------------------------------------------------------------------------
#                             draw_xr_dataarray_on_map
# -------------------------------------------------------------------------------


def plot_xr_dataarray_on_map(
    data: xr.Dataset,
    band: str,
    timestep: int = 0,
    map: folium.Map = None,
    cmap: str = "viridis",
) -> folium.Map:
    # Extract the DataArray for the specified band and timestep
    selected_data = data[band].isel(t=timestep)

    # Set up transformer from SWEREF99 to WGS84
    transformer = Transformer.from_crs(
        "EPSG:3006", "EPSG:4326", always_xy=True
    )  # Replace EPSG:3006 if using a different SWEREF zone

    # Transform the SWEREF bounds to WGS84
    min_lon, min_lat = transformer.transform(data.x.values.min(), data.y.values.min())
    max_lon, max_lat = transformer.transform(data.x.values.max(), data.y.values.max())
    bounds = [[min_lat, min_lon], [max_lat, max_lon]]  # south-west to north-east

    # Also, transform the central starting point for the map
    center_lon, center_lat = transformer.transform(
        (data.x.values.min() + data.x.values.max()) / 2,
        (data.y.values.min() + data.y.values.max()) / 2,
    )

    # Create a new map centered on the transformed coordinates if none is provided
    if map is None:
        map = folium.Map(
            location=[center_lat, center_lon], zoom_start=9
        )  # Adjust zoom as needed

    # Normalize and create colormap
    norm = Normalize(vmin=float(selected_data.min()), vmax=float(selected_data.max()))
    colormap = plt.get_cmap(cmap)  # Use string to get colormap

    # Convert the data to RGBA for plotting
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        normalized_data = (norm(selected_data.values) * 255).astype(np.uint8)
    img = plt.cm.ScalarMappable(norm=norm, cmap=colormap).to_rgba(
        normalized_data, bytes=True
    )

    # Create an image overlay
    folium.raster_layers.ImageOverlay(
        image=img,
        bounds=bounds,
        opacity=0.7,
        interactive=True,
        cross_origin=False,
        zindex=1,
    ).add_to(map)

    return map


# -----------------------------------------------------------------------------
#                               load_geotiff_as_xarray
# -----------------------------------------------------------------------------
def load_geotiff_as_xarray(binary_data: bytes, band_ix: int = 1) -> xr.DataArray:
    """
    Load a GeoTIFF file from binary data and return it as an xarray DataArray.

    Parameters:
    - binary_data (bytes): The binary content of a GeoTIFF file.
    - band_ix (int): The index of the band to read. Defaults to 1.

    Returns:
    - xr.DataArray: The xarray DataArray loaded from the GeoTIFF file.

    Raises:
    - Exception: If the data cannot be loaded as a GeoTIFF file, possibly due to
      it being a different format such as ZIP or NetCDF.
    """
    try:
        # Open the GeoTIFF from binary data
        with rasterio.open(BytesIO(binary_data)) as im:
            # Read the specified band into a numpy array
            array = im.read(band_ix)

            # Handle the transform and CRS, providing defaults if not available
            transform = (
                im.transform if im.transform is not None else rasterio.Affine.identity()
            )
            crs = im.crs.to_string() if im.crs is not None else "unknown"

            # Generate coordinates based on the transform
            y_coords = np.arange(array.shape[0]) * transform[4] + transform[5]
            x_coords = np.arange(array.shape[1]) * transform[0] + transform[2]

            # Convert the numpy array into an xarray DataArray
            data_array = xr.DataArray(
                array,
                dims=["y", "x"],
                coords={
                    "y": y_coords,
                    "x": x_coords,
                },
                attrs={
                    "transform": transform,
                    "crs": crs,
                    "description": "Loaded from binary data",
                },
            )

        return data_array

    except Exception as e:
        # Catch and re-raise any exception with a more specific error message
        raise Exception(
            f"Failed to load GeoTIFF. Ensure the data is a valid GeoTIFF format, not a ZIP or NetCDF file. Error details: {str(e)}"
        ) from e


# -----------------------------------------------------------------------------
#                               load_netcdf_as_xarray
# -----------------------------------------------------------------------------
def load_netcdf_as_xarray(binary_data: bytes) -> xr.Dataset:
    """
    Load a NetCDF file from binary data and return it as an xarray Dataset.

    Parameters:
    - binary_data (bytes): The binary content of a NetCDF file.

    Returns:
    - xr.Dataset: The xarray Dataset loaded from the NetCDF file.

    Raises:
    - Exception: If the data cannot be loaded as a NetCDF file, possibly due to
      it being a different format such as ZIP or GeoTIFF.
    """
    try:
        # Try to open the dataset using xarray
        dataset = xr.open_dataset(BytesIO(binary_data))
        return dataset
    except Exception as e:
        # Catch and re-raise any exception with a more specific error message
        raise Exception(
            f"Failed to load NetCDF. Ensure the data is a valid NetCDF format, not a ZIP or GeoTIFF file. Error details: {str(e)}"
        ) from e


# -----------------------------------------------------------------------------
#                       show_binary_array_thumbnail
# -----------------------------------------------------------------------------
def show_binary_array_thumbnail(array, title=None) -> None:
    """
    Display a binary thumbnail of a 2D array, showing the binary representation of 
    the first three columns of each row. The function is useful for visualizing
    the binary structure of data in a more human-readable format.

    Parameters:
    - array: 2D numpy array to be visualized. Each element of the array is 
             expected to be a numeric value.
    - title: Optional string to be displayed as the title above the thumbnail.
    """
    if title:
        w = 32 * 3 + 6  # Total width based on 32-bit columns and spaces between them
        s = int((w - len(title)) / 2)
        p = ''.join(['-'] * s)
        title_row = f"{p}{title}{p}"
        print(title_row)
        print()
    
    # Create the index row with bits counting from right (LSB) to left (MSB)
    index_row_top = "  ".join(["33222222222211111111110000000000"] * 3)
    print(index_row_top)
    
    index_row_bottom = "  ".join(["10987654321098765432109876543210"] * 3)
    print(index_row_bottom)
    print()

    # Print each row of binary representations
    for row in array[:8]:
        for col in row[:3]:  # Only take the first 3 columns for display
            if not np.isnan(col):  # Only process non-NaN values
                # Truncate the float64 value to an integer
                truncated_value = math.trunc(col)
                # Convert the truncated integer to a 32-bit binary representation
                s = f"{truncated_value:032b}".replace('0', '_')
                print(s, end="   ")
            else:
                print("N".ljust(35), end="")  # Display 'N' for NaN values
        print()  # Newline after each row

    if title:
        print()
        print(''.join(['-'] * w))


# -----------------------------------------------------------------------------
#                        show_binary_image_thumbnail
# -----------------------------------------------------------------------------
def show_binary_image_thumbnail(
    image_list: np.ndarray, title=None, band=1, index=0
) -> None:
    """
    Display a binary thumbnail of a specific band from an image in a 4D numpy array.

    Parameters:
    - image_list: A 4D numpy array of shape (timestep, band, x, y).
    - title: Optional string to be displayed as the title above the thumbnail.
    - band: The band number to read from the raster image (default is 1).
    - index: The index of the time step in the image_list to be visualized (default is 0).

    The function reads the specified band of the selected image, converts it to
    a binary thumbnail, and displays the first three columns of each row as
    32-bit binary values using `show_binary_array_thumbnail`.
    """
    array = image_list[index, band - 1]  # Access the (timestep, band, x, y) data

    # Display the binary thumbnail of the array
    show_binary_array_thumbnail(array, title)

