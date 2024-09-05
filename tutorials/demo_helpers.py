import os
import io
import tarfile
import tempfile
from pathlib import Path
import matplotlib.pyplot as plt
import rasterio.plot
import yaml
import rasterio
import xarray as xr
import numpy as np
from io import BytesIO

# -----------------------------------------------------------------------------
#                               showfig
# -----------------------------------------------------------------------------
def showfig(data,flag = None,figsize=(5,5),ax=None):
    colorbar = ax is None
    ax = ax or plt.figure(figsize=figsize)
    if flag:
        img = (data & flag) == flag
    else:
        img = data
    plt.imshow(img)
    if colorbar:
        plt.colorbar()
        
# Normalize the data to the range 0-255 for display
def normalize(array):
    array_min, array_max = np.percentile(array, (1, 99))  # Clip values between 1st and 99th percentile
    array = np.clip(array, array_min, array_max)  # Clip the extreme values
    return ((array - array_min) / (array_max - array_min) * 255).astype(np.uint8)
# -----------------------------------------------------------------------------
#                         show_single_result
#
# Visualizes every band of a single Geotiff result from OpenEO as a separate image
# -----------------------------------------------------------------------------
def show_single_result(image_data, colormap, is_ndvi=False, title=None):
    '''
    image_data - a single image file response from openeo
    is_ndvi    - set this to true if you have done NDVI calculations
                 (sets a nicer color map)
    '''
    
    # Check if the image data is empty
    if not image_data:
        print("No image data available.")
        return
    # Check if the dataset is empty or has valid data
    filelike = io.BytesIO(image_data)
    im = rasterio.open(filelike)
    if im.count == 0:
        print("The dataset is empty.")
        return
    for i in range(1, im.count + 1):
        fig, ax = plt.subplots(figsize=(12,12))
        b = im.read(i)
        b_norm = normalize(b)   
        if is_ndvi:
            rasterio.plot.show(b_norm, ax=ax, cmap='RdYlGn')
        else:
            rasterio.plot.show(b_norm, ax=ax, cmap=colormap)      
        #Set title if not provided
        title = title or f'Created: {im.tags().get("datetime_from_dim", "Unknown Date")}'
        plt.title(title)
    
    return [im]
# -----------------------------------------------------------------------------
#                           show_zipped_results
# Visualizes each result of multiple zipped Geotiffs from OpenEO in a single plot. 
# The first band of each result is visualized as a subplot.
#
# -----------------------------------------------------------------------------
def show_zipped_results(image_data, colormap, is_ndvi=False,title=None):
    '''
    image_data - a single image file response from openeo
    is_ndvi    - set this to true if you have done NDVI calculations
                 (sets a nicer color map)
    EXAMPLE             
    res=connection.load_collection(s2.s2_msi_l2a,
                         spatial_extent=s2.bbox.karlstad_mini_land,
                         temporal_extent=s2.timespans.five_images,
                        bands=['b08','b04']
                        )
    image_data = res.download(format="gtiff")
    show_single_result(image_data)
    
    
    '''
    images = []
    file_like_object = io.BytesIO(image_data)
    
    with tempfile.TemporaryDirectory() as tmpdirname:

        # Step 2: Open the tar.gz file
        with tarfile.open(fileobj=file_like_object, mode="r:gz") as tar:
            # Step 3: Extract all the contents into a specific directory
            tar.extractall(tmpdirname)
            
            if os.path.isdir(tmpdirname):
                tmpdirname = f"{tmpdirname}/{os.listdir(tmpdirname)[0]}"
                
            
            
            image_types = [".tif"]

            ifnames = [ifname for ifname in sorted(os.listdir(tmpdirname))
                       if  any(image_type in ifname for image_type in image_types)]
            
            columns = 2  # For example, adjust based on your preference and screen size

            # Calculate the required number of rows to fit all images
            rows = len(ifnames) // columns + (len(ifnames) % columns > 0)

            # Create subplots
            fig, axs = plt.subplots(rows, columns, figsize=(12,12))
            for idx, ifname in enumerate(ifnames):
                fname = f"{tmpdirname}/{ifname}"
                src = rasterio.open(fname)
                images.append(src)
                if rows > 1:
                    ax = axs[idx // columns][idx % columns]
                else:
                    ax = axs[idx]

                if is_ndvi:
                    rasterio.plot.show(src, ax=ax, cmap='RdYlGn')
                else:
                    src_norm = normalize(src.read(1))
                    rres = rasterio.plot.show(src_norm, transform=src.transform, ax=ax, cmap=colormap)
                    im = rres.get_images()[0]
                    fig.colorbar(im, ax=ax, shrink=0.35, aspect=10)
                title = title or f'Created:{src.tags()["datetime_from_dim"]}'
                ax.set_title(title)
        return images
# -----------------------------------------------------------------------------
#                               show_result
# -----------------------------------------------------------------------------
def show_result(image_data, colormap='viridis', is_ndvi=False, title=None):
    try:
        return show_single_result(image_data, colormap, is_ndvi, title)
    except Exception as e:
        pass
    return show_zipped_results(image_data, colormap, is_ndvi, title)

# -----------------------------------------------------------------------------
#                               get_s3_wqsf_flags
# -----------------------------------------------------------------------------
def get_s3_wqsf_flags():
    '''
    You can get these flags from get_collections, but this is a shortcut for
    training purposes.
    
    '''
    wqsf_flags = {}
    here = Path(__file__).parent
    with open(f"{here}/s3_olci_l2wfr.odc-product.yaml", 'r') as stream:
        s3_meta = yaml.safe_load(stream)
      
        for m in s3_meta['measurements']:
            if 'wqsf' in m['name']:
                bits = (m['flags_definition']['data']['values'])
                bitmap = {}
                for b in bits.keys():
                    bitmap[bits[b]] = b 
                    
                wqsf_flags[m['name']] = bitmap
    return wqsf_flags


# -----------------------------------------------------------------------------
#                                plot_xr_DataArray
# -----------------------------------------------------------------------------
import warnings

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from geopandas.geodataframe import GeoDataFrame

RasterCube = xr.DataArray

def plot_xr_dataarray_contact_copy(
    data: xr.DataArray,
    cols: int = 4,
    skip_nans=True,
    figsize=(10, 15),
    cmap='viridis'  
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
                fraction = round(100* (valid_pixels / total_pixels),2)
                ax.set_title(f"{str(data.t.values[i]).split('T')[0]} Valid={fraction}% Mean={round(mean_value,2)}")
                #print(im.count())
                ax.axis("on")  # Or 'off' depending on your preference
            else:
                ax.set_title(f"{str(data.t.values[i]).split('T')[0]} All NaNs")
                
        else:
            ax.axis('off')  # Hide unused subplots

    plt.show()

# -------------------------------------------------------------------------------
#                             draw_xr_dataarray_on_map
# -------------------------------------------------------------------------------
import folium
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import xarray as xr
from pyproj import Transformer
import warnings
def plot_xr_dataarray_on_map(data: xr.Dataset, band: str, timestep: int = 0, map: folium.Map = None, cmap: str = 'viridis') -> folium.Map:
    # Extract the DataArray for the specified band and timestep
    selected_data = data[band].isel(t=timestep)
    
    # Set up transformer from SWEREF99 to WGS84
    transformer = Transformer.from_crs("EPSG:3006", "EPSG:4326", always_xy=True)  # Replace EPSG:3006 if using a different SWEREF zone

    # Transform the SWEREF bounds to WGS84
    min_lon, min_lat = transformer.transform(data.x.values.min(), data.y.values.min())
    max_lon, max_lat = transformer.transform(data.x.values.max(), data.y.values.max())
    bounds = [[min_lat, min_lon], [max_lat, max_lon]]  # south-west to north-east

    # Also, transform the central starting point for the map
    center_lon, center_lat = transformer.transform((data.x.values.min() + data.x.values.max()) / 2, 
                                                   (data.y.values.min() + data.y.values.max()) / 2)
    
    # Create a new map centered on the transformed coordinates if none is provided
    if map is None:
        map = folium.Map(location=[center_lat, center_lon], zoom_start=9)  # Adjust zoom as needed

    # Normalize and create colormap
    norm = Normalize(vmin=float(selected_data.min()), vmax=float(selected_data.max()))
    colormap = plt.get_cmap(cmap)  # Use string to get colormap

    # Convert the data to RGBA for plotting
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        normalized_data = (norm(selected_data.values) * 255).astype(np.uint8)
    img = plt.cm.ScalarMappable(norm=norm, cmap=colormap).to_rgba(normalized_data, bytes=True)

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

import xarray as xr
import rasterio
from io import BytesIO
import numpy as np

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
            transform = im.transform if im.transform is not None else rasterio.Affine.identity()
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
                    "description": "Loaded from binary data"
                }
            )
        
        return data_array

    except Exception as e:
        # Catch and re-raise any exception with a more specific error message
        raise Exception(f"Failed to load GeoTIFF. Ensure the data is a valid GeoTIFF format, not a ZIP or NetCDF file. Error details: {str(e)}") from e

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
        raise Exception(f"Failed to load NetCDF. Ensure the data is a valid NetCDF format, not a ZIP or GeoTIFF file. Error details: {str(e)}") from e


# -----------------------------------------------------------------------------
#                       show_binary_array_thumbnail
# -----------------------------------------------------------------------------
import math
import rasterio.io as rio

def show_binary_array_thumbnail(array, title=None) -> None:
    """
    Display a thumbnail of a 2D array, showing the binary representation of 
    the first three columns of each row. The function is useful for visualizing
    the binary structure of data in a more human-readable format.

    Parameters:
    - array: 2D numpy array to be visualized. Each element of the array is 
             expected to be a numeric value.
    - title: Optional string to be displayed as the title above the thumbnail.

    The output displays a header with bit positions and the first three 
    columns of each row as 32-bit binary values, with `0` replaced by `_` 
    for better visual distinction.
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
            # Truncate the float64 value to an integer
            truncated_value = math.trunc(col)
            # Convert the truncated integer to a 32-bit binary representation
            s = f"{truncated_value:032b}".replace('0', '_')
            print(s, end="   ")
        # Print a newline after each row
        print()
    
    if title:
        print()
        print(''.join(['-'] * w))

# -----------------------------------------------------------------------------
#                        show_binary_image_thumbnail
# -----------------------------------------------------------------------------
def show_binary_image_thumbnail(image_list: list[rio.DatasetReader], title=None, band=1, index=0) -> None:
    """
    Display a binary thumbnail of a specific band from an image in a list of 
    raster images.

    Parameters:
    - image_list: A list of rasterio DatasetReader objects representing the images.
    - title: Optional string to be displayed as the title above the thumbnail.
    - band: The band number to read from the raster image (default is 1).
    - index: The index of the image in the image_list to be visualized (default is 0).

    The function reads the specified band of the selected image, converts it to 
    a binary thumbnail, and displays the first three columns of each row as 
    32-bit binary values using `show_binary_array_thumbnail`.
    """
    array = image_list[index].read(band)  # Read the specified band from the selected image
    show_binary_array_thumbnail(array, title)  # Display the binary thumbnail of the array

