#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 12:05:05 2025

@author: agastraa
"""

import numpy as np
from netCDF4 import Dataset
import os

def nfcheck(status):
    if status == 0:
        return
    print('Error:', status)
    raise RuntimeError('NetCDF error')

def get_nc_vals(fname, var):
    try:
        with Dataset(fname, 'r') as nc_file:
            lat_name = 'latitude'
            lon_name = 'longitude'
            
            if 'lat_0' in nc_file.variables:
                lat_name = 'lat_0'
                lon_name = 'lon_0'
            elif 'lat_3' in nc_file.variables:
                lat_name = 'lat_3'
                lon_name = 'lon_3'
            elif 'lat' in nc_file.variables:
                lat_name = 'lat'
                lon_name = 'lon'
            
            nc_lats = nc_file.variables[lat_name][:]
            nc_lons = nc_file.variables[lon_name][:]
            
            nc_vals = nc_file.variables[var][:]
            
            nc_grid = {
                'xspan': len(nc_lons),
                'yspan': len(nc_lats),
                'xmin': np.min(nc_lons),
                'ymin': np.min(nc_lats),
                'xmax': np.max(nc_lons),
                'ymax': np.max(nc_lats),
                'xres': (np.max(nc_lons) - np.min(nc_lons)) / len(nc_lons),
                'yres': (np.max(nc_lats) - np.min(nc_lats)) / len(nc_lats),
                'reversed': nc_lats[0] > nc_lats[1]
            }
            
            return nc_vals, nc_grid
    except Exception as e:
        print(f"Error processing {fname}: {e}")
        return None, None

def get_nc_latlon(fname, var, lat, lon):
    try:
        with Dataset(fname, 'r') as nc_file:
            lat_name = 'latitude'
            lon_name = 'longitude'
            
            if 'lat_0' in nc_file.variables:
                lat_name = 'lat_0'
                lon_name = 'lon_0'
            elif 'lat_3' in nc_file.variables:
                lat_name = 'lat_3'
                lon_name = 'lon_3'
            elif 'lat' in nc_file.variables:
                lat_name = 'lat'
                lon_name = 'lon'
            
            nc_lats = nc_file.variables[lat_name][:]
            nc_lons = nc_file.variables[lon_name][:]
            
            nc_grid = {
                'xspan': len(nc_lons),
                'yspan': len(nc_lats),
                'xmin': np.min(nc_lons),
                'ymin': np.min(nc_lats),
                'xmax': np.max(nc_lons),
                'ymax': np.max(nc_lats),
                'xres': (np.max(nc_lons) - np.min(nc_lons)) / len(nc_lons),
                'yres': (np.max(nc_lats) - np.min(nc_lats)) / len(nc_lats),
                'reversed': nc_lats[0] > nc_lats[1]
            }
            
            x = int((lon - nc_grid['xmin']) / nc_grid['xres'])
            y = int((lat - nc_grid['ymin']) / nc_grid['yres'])
            
            if 0 <= x < nc_grid['xspan'] and 0 <= y < nc_grid['yspan']:
                return nc_file.variables[var][y, x]
            else:
                return None
    except Exception as e:
        print(f"Error processing {fname}: {e}")
        return None



def initialize_nc_outfile(out_base, map_grid):
    print("Initializing netcdf format output file . . .")

    # Create the netCDF file
    ncfname = f"{out_base.strip()}_rpmaps.nc"
    ncid = Dataset(ncfname, 'w', format='NETCDF4')

    # Define dimensions
    lat_dimid = ncid.createDimension('latitude', map_grid['yspan'])
    lon_dimid = ncid.createDimension('longitude', map_grid['xspan'])

    # Define variables
    lat_varid = ncid.createVariable('latitude', np.float32, ('latitude',))
    lon_varid = ncid.createVariable('longitude', np.float32, ('longitude',))

    # Add attributes
    lat_varid.units = 'degrees_north'
    lon_varid.units = 'degrees_east'

    # Allocate arrays for latitudes and longitudes
    lats = np.empty(map_grid['yspan'], dtype=np.float32)
    lons = np.empty(map_grid['xspan'], dtype=np.float32)

    # Fill latitudes and longitudes based on the map grid
    if map_grid['reversed']:
        print("Writing reversed format")
        lats = map_grid['ymax'] - np.arange(1, map_grid['yspan'] + 1) * map_grid['yres']
    else:
        lats = map_grid['ymin'] + np.arange(1, map_grid['yspan'] + 1) * map_grid['yres']

    lons = map_grid['xmin'] + np.arange(1, map_grid['xspan'] + 1) * map_grid['xres']

    # Write latitudes and longitudes to the netCDF file
    lat_varid[:] = lats
    lon_varid[:] = lons

    # Close the netCDF file
    ncid.close()


def write_nc_grid(varname, varunits, out_base, nc_varname, nc_varunits, rp_outvals):
    print(f"Writing {varname} in netcdf format . . .")

    src_varname = f"{varname}_{nc_varname}"

    ncfname = f"{out_base.strip()}_rpmaps.nc"

    # Open the NetCDF file for writing
    ncid = Dataset(ncfname, 'a')  # 'a' mode opens the file for appending

    # Define dimensions if they don't already exist
    if 'latitude' not in ncid.dimensions:
        lat_dimid = ncid.createDimension('latitude', len(rp_outvals))
    else:
        lat_dimid = ncid.dimensions['latitude']

    if 'longitude' not in ncid.dimensions:
        lon_dimid = ncid.createDimension('longitude', len(rp_outvals[0]))
    else:
        lon_dimid = ncid.dimensions['longitude']

    # Redefine the NetCDF file
    ncid.setncattr('redef', True)

    # Define variables
    if varname not in ncid.variables:
        targ_varid = ncid.createVariable(varname, np.float32, ('longitude', 'latitude'))
        targ_varid.units = varunits

    if src_varname not in ncid.variables:
        src_varid = ncid.createVariable(src_varname, str, ())
        src_varid.units = nc_varunits

    # End definition mode
    ncid.setncattr('redef', False)

    # Write data to the variables
    ncid.variables[varname][:] = rp_outvals
    # ncid.variables[src_varname][:] = nc_varname  # Uncomment if you need to write the source variable name

    # Close the NetCDF file
    ncid.close()

    print('done')





def load_ncfile_list(nc_file_list, map_grid, user_grid, debug=False):
    def nfcheck(status):
        if status != "No Error":
            raise Exception(f"NetCDF error: {status}")

    def num_records(file_path):
        with open(file_path, 'r') as f:
            return sum(1 for line in f if line.strip() and not line.startswith('#'))

    def julian_date(mm, dd, yy, hh):
        # Placeholder for the Julian date calculation
        return (yy - 2000) * 365 + mm * 30 + dd + hh / 24.0

    def caldat(julian_day, yy, mm, dd, hh):
        # Placeholder for converting Julian date to calendar date
        yy = 2000 + julian_day // 365
        mm = (julian_day % 365) // 30
        dd = (julian_day % 365) % 30
        hh = (julian_day * 24) % 24

    if not os.path.exists(nc_file_list):
        return

    print(f"Scanning {nc_file_list}")

    maxpts = num_records(nc_file_list)
    nc_files = [None] * maxpts
    nc_jday = [None] * maxpts
    valid_nc = [True] * maxpts
    errors = 0

    num_ncfiles = 0
    with open(nc_file_list, 'r') as f:
        for line in f:
            pbuffy = line.strip()
            if not pbuffy or pbuffy.startswith('#'):
                continue
            num_ncfiles += 1
            nc_files[num_ncfiles - 1] = pbuffy
            if debug:
                print(pbuffy)
            if not os.path.exists(pbuffy):
                print(f"Warning: {pbuffy} does not exist!")
                num_ncfiles -= 1
                errors += 1
                continue

            try:
                ncid = Dataset(pbuffy, 'r')
            except Exception as e:
                print(f"Warning: {pbuffy} may be corrupt; skipping.")
                num_ncfiles -= 1
                errors += 1
                continue

            DLAT_NAME = 'latitude'
            DLON_NAME = 'longitude'
            LAT_NAME = 'latitude'
            LON_NAME = 'longitude'
            if 'lat_0' in ncid.variables:
                DLAT_NAME = 'lat_0'
                DLON_NAME = 'lon_0'
            if 'lat_3' in ncid.variables:
                DLAT_NAME = 'lat_3'
                DLON_NAME = 'lon_3'
                LAT_NAME = 'lat_3'
                LON_NAME = 'lon_3'
            if 'lat' in ncid.variables:
                DLAT_NAME = 'lat'
                DLON_NAME = 'lon'
                LAT_NAME = 'lat'
                LON_NAME = 'lon'

            lat_dimid = ncid.dimensions[DLAT_NAME]
            lon_dimid = ncid.dimensions[DLON_NAME]
            nLats = len(lat_dimid)
            nLons = len(lon_dimid)

            if num_ncfiles == 1:
                nc_nlats = nLats
                nc_nlons = nLons

                lat_varid = ncid.variables[LAT_NAME]
                lon_varid = ncid.variables[LON_NAME]
                nc_lats = lat_varid[:]
                nc_lons = lon_varid[:]
                nc_grid = {
                    'xspan': nc_nlons,
                    'yspan': nc_nlats,
                    'xmin': np.min(nc_lons),
                    'ymin': np.min(nc_lats),
                    'xmax': np.max(nc_lons),
                    'ymax': np.max(nc_lats),
                    'reversed': nc_lats[0] > nc_lats[1],
                    'xres': (np.max(nc_lons) - np.min(nc_lons)) / nc_nlons,
                    'yres': (np.max(nc_lats) - np.min(nc_lats)) / nc_nlats
                }
            else:
                if nc_nlats != nLats or nc_nlons != nLons:
                    identical_grid = False

            if user_grid:
                lat_varid = ncid.variables[LAT_NAME]
                lon_varid = ncid.variables[LON_NAME]
                nc_nlats = nLats
                nc_nlons = nLons
                nc_lats = lat_varid[:]
                nc_lons = lon_varid[:]

                if np.min(nc_lons) > map_grid['xmax'] or np.max(nc_lons) < map_grid['xmin']:
                    valid_nc[num_ncfiles - 1] = False
                if np.min(nc_lats) > map_grid['ymax'] or np.max(nc_lats) < map_grid['ymin']:
                    valid_nc[num_ncfiles - 1] = False

            yy = 1
            mm = 1
            dd = 1
            hh = 0.0
            if 'julian_day' in ncid.variables:
                varid = ncid.variables['julian_day']
                nc_jday[num_ncfiles - 1] = varid[:]
                caldat(nc_jday[num_ncfiles - 1], yy, mm, dd, hh)
            if 'BeginDate' in ncid.ncattrs():
                daystr = ncid.getncattr('BeginDate')
                yy, mm, dd = map(int, daystr.split('-'))
                hh = 0.0
                nc_jday[num_ncfiles - 1] = julian_date(mm, dd, yy, 0.0)
            if 'dtg' in ncid.ncattrs():
                daystr = ncid.getncattr('dtg')
                mm, dd, yy = map(int, daystr.split('-'))
                hh = 0.0
                nc_jday[num_ncfiles - 1] = julian_date(mm, dd, yy, 0.0)
                if debug:
                    print(f"Loaded {yy}-{mm}-{dd} {hh}")

            ncid.close()

    print(f"Scanned {num_ncfiles} netcdf files, {errors} errors.")
    if identical_grid:
        print("  Files appear to be using the same grid.")
    else:
        print("  Notice: Files DO NOT appear to be using the same grid.")

# Example usage
nc_file_list = 'example_nc_file_list.txt'
map_grid = {
    'xspan': 360,
    'yspan': 180,
    'xmin': -180.0,
    'ymin': -90.0,
    'xmax': 180.0,
    'ymax': 90.0,
}
user_grid = True
identical_grid = True

load_ncfile_list(nc_file_list, map_grid, user_grid, debug=True)