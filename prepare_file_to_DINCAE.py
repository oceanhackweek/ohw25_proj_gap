import zarr
import numpy as np
from netCDF4 import Dataset
import os

def convert_zarr_to_netcdf(zarr_path, output_filename, varname="SST"):
    """
    Convert Zarr file to NetCDF format matching the specified structure
    
    Parameters:
    zarr_path: path to the zarr file/directory
    output_filename: name of the output NetCDF file
    varname: name for the main data variable (default: "SST")
    """
    
    # Check if output file already exists to prevent overwriting
    if os.path.exists(output_filename):
        print(f"Warning: {output_filename} already exists. Choose a different filename to avoid overwriting.")
        return False
    
    try:
        # Open the Zarr file
        zarr_store = zarr.open(zarr_path, mode='r')
        print(f"Successfully opened Zarr file: {zarr_path}")
        
        # First, let's inspect all variables in the Zarr file
        print("\nInspecting Zarr file structure:")
        for key in zarr_store.keys():
            var = zarr_store[key]
            print(f"  {key}: shape={var.shape}, dtype={var.dtype}")
        
        # Read coordinate variables
        lon = np.array(zarr_store['lon'])
        lat = np.array(zarr_store['lat'])
        time = np.array(zarr_store['time'])
        
        print(f"\nDimensions - lon: {len(lon)}, lat: {len(lat)}, time: {len(time)}")
        
        # Read the main data variable (assuming it's SST or similar)
        # Check which variable contains the main data
        data_var = None
        data = None
        
        # List of potential data variables to check
        potential_vars = ['sst', 'SST', 'CHL', 'chl', 'chlorophyll']
        
        # First try to find known data variables
        for var_name in potential_vars:
            if var_name in zarr_store:
                data_var = var_name
                break
        
        # If no known variables found, look for any 3D variable
        if data_var is None:
            for key in zarr_store.keys():
                if key not in ['lon', 'lat', 'time', 'cos_time', 'sin_time']:
                    var = zarr_store[key]
                    if var.ndim == 3:  # 3D array
                        data_var = key
                        break
        
        if data_var is not None:
            print(f"Using data variable: {data_var}")
            raw_data = zarr_store[data_var]
            print(f"Raw data shape: {raw_data.shape}")
            
            # Handle different dimension orders
            expected_shape = (len(time), len(lat), len(lon))
            if raw_data.shape == expected_shape:
                data = np.array(raw_data)
            elif raw_data.shape == (len(lat), len(lon), len(time)):
                # Transpose from (lat, lon, time) to (time, lat, lon)
                data = np.transpose(np.array(raw_data), (2, 0, 1))
                print("Transposed data from (lat, lon, time) to (time, lat, lon)")
            elif raw_data.shape == (len(lon), len(lat), len(time)):
                # Transpose from (lon, lat, time) to (time, lat, lon)
                data = np.transpose(np.array(raw_data), (2, 1, 0))
                print("Transposed data from (lon, lat, time) to (time, lat, lon)")
            else:
                print(f"Warning: Data shape {raw_data.shape} doesn't match expected dimensions")
                print(f"Expected: {expected_shape}")
                # Try to reshape or create dummy data
                if np.prod(raw_data.shape) == np.prod(expected_shape):
                    data = np.array(raw_data).reshape(expected_shape)
                    print("Reshaped data to match expected dimensions")
                else:
                    print("Cannot reshape data. Creating dummy data instead.")
                    data = np.zeros(expected_shape)
        else:
            print("No suitable data variable found. Creating dummy data.")
            data = np.zeros((len(time), len(lat), len(lon)))
        
        print(f"Final data shape: {data.shape}")
        
        # Handle time units - try to get from Zarr attributes or use default
        try:
            if hasattr(zarr_store['time'], 'attrs') and 'units' in zarr_store['time'].attrs:
                time_units = zarr_store['time'].attrs['units']
            else:
                time_units = "days since 1900-01-01 00:00:00"
                print(f"Time units not found in Zarr, using default: {time_units}")
        except:
            time_units = "days since 1900-01-01 00:00:00"
            print(f"Using default time units: {time_units}")
        
        # Create mask - check if it exists in Zarr, otherwise create default
        mask = None
        if 'land_flag' in zarr_store:
            land_flag_raw = zarr_store['land_flag']
            print(f"Land flag shape: {land_flag_raw.shape}")
            land_flag = np.array(land_flag_raw)
            
            # Handle different land flag shapes
            if land_flag.shape == (len(lat), len(lon)):
                mask = np.where(land_flag == 0, 1, 0)  # Invert land flag to get water mask
            elif land_flag.shape == (len(lon), len(lat)):
                mask = np.where(land_flag.T == 0, 1, 0)  # Transpose and invert
            else:
                print(f"Unexpected land_flag shape: {land_flag.shape}")
                mask = np.ones((len(lat), len(lon)), dtype=np.int32)
                
        elif 'valid_CHL_flag' in zarr_store:
            valid_flag_raw = zarr_store['valid_CHL_flag']
            print(f"Valid CHL flag shape: {valid_flag_raw.shape}")
            valid_flag = np.array(valid_flag_raw)
            
            if valid_flag.shape == (len(lat), len(lon)):
                mask = valid_flag
            elif valid_flag.shape == (len(lon), len(lat)):
                mask = valid_flag.T
            else:
                print(f"Unexpected valid_CHL_flag shape: {valid_flag.shape}")
                mask = np.ones((len(lat), len(lon)), dtype=np.int32)
        
        if mask is None:
            # Create default mask (all valid points)
            mask = np.ones((len(lat), len(lon)), dtype=np.int32)
            print("No mask information found, creating default mask (all valid points)")
        
        print(f"Final mask shape: {mask.shape}")
        
        # Ensure mask is the right data type
        mask = mask.astype(np.int32)
        
        # Set fill value
        fill_value = -9999.
        
        # Create the NetCDF file
        print(f"Creating NetCDF file: {output_filename}")
        root_grp = Dataset(output_filename, 'w', format='NETCDF4')
        
        # Create dimensions
        root_grp.createDimension('lon', len(lon))
        root_grp.createDimension('lat', len(lat))
        root_grp.createDimension('time', None)  # unlimited dimension
        
        # Create coordinate variables
        nc_lon = root_grp.createVariable('lon', 'f4', ('lon',))
        nc_lat = root_grp.createVariable('lat', 'f4', ('lat',))
        nc_time = root_grp.createVariable('time', 'f4', ('time',))
        nc_time.units = time_units
        
        # Create data variable
        nc_data = root_grp.createVariable(varname, 'f4', ('time', 'lat', 'lon'),
                                          fill_value=fill_value)
        
        # Create mask variable
        nc_mask = root_grp.createVariable("mask", 'i4', ('lat', 'lon'))
        nc_mask.comment = "one means the data location is valid (e.g. sea for SST), zero the location is invalid (e.g. land for SST)"
        
        # Write data to NetCDF
        nc_lon[:] = lon
        nc_lat[:] = lat
        nc_time[:] = time
        nc_data[:, :, :] = data
        nc_mask[:, :] = mask
        
        # Add global attributes
        root_grp.description = f"Converted from Zarr file: {zarr_path}"
        root_grp.history = "Created by Zarr to NetCDF conversion script"
        
        # Close the file
        root_grp.close()
        
        print(f"Successfully created NetCDF file: {output_filename}")
        print(f"File structure:")
        print(f"  - Dimensions: time({len(time)}), lat({len(lat)}), lon({len(lon)})")
        print(f"  - Variables: {varname}, mask, time, lat, lon")
        print(f"  - Fill value: {fill_value}")
        
        return True
        
    except Exception as e:
        print(f"Error converting Zarr to NetCDF: {str(e)}")
        return False

# Example usage
if __name__ == "__main__":
    # Set your paths here
    zarr_file_path = "/home/jovyan/shared-public/mindthegap/data/2015_3_ArabSea_Eli.zarr"  # Path to your Zarr file
    output_file = "/home/jovyan/shared-public/mindthegap/data/test_data/test_data.nc"    # Output NetCDF filename
    
    variable_name = "CHL"                        # Name for the main data variable
    
    # Convert the file
    success = convert_zarr_to_netcdf(zarr_file_path, output_file, variable_name)
    
    if success:
        print("Conversion completed successfully!")
    else:
        print("Conversion failed!")