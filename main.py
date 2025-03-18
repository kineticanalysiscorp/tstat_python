#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 12:29:06 2025

@author: agastraa
"""

import os
import argparse
import numpy as np
from scipy.stats import weibull_min, norm
from datetime import datetime
from netCDF4 import Dataset
import time
import csv
import ncsubs

def main():
    parser = argparse.ArgumentParser(description='TAOS(tm) Hazard Frequency Analysis System')
    parser.add_argument('-fitext', type=str, help='Fit external file')
    parser.add_argument('-table', type=str, help='SQL table name')
    parser.add_argument('-thres', type=float, help='Master threshold')
    parser.add_argument('-ncscale', type=float, help='NC scale')
    parser.add_argument('-pre', type=str, help='Parameter prefix')
    parser.add_argument('-numvals', type=int, help='Number of values')
    parser.add_argument('-ncfiles', type=str, help='NC file list')
    parser.add_argument('-ncvar', type=str, help='NC variable name')
    parser.add_argument('-points', type=str, help='Point file list')
    parser.add_argument('-monthspi', nargs=4, type=int, help='Month SPI parameters')
    parser.add_argument('-spi', type=float, help='SPI query value')
    parser.add_argument('-mfv', type=float, help='Max fit values')
    parser.add_argument('-march', type=float, help='March size')
    parser.add_argument('-eval', type=str, help='Evaluation map file')
    parser.add_argument('-rpmaps', type=str, help='RP map list')
    parser.add_argument('-annualsum', action='store_true', help='Annual sum aggregation')
    parser.add_argument('-annualmax', action='store_true', help='Annual max aggregation')
    parser.add_argument('-monthmax', type=int, help='Month max aggregation')
    parser.add_argument('-monthsum', type=int, help='Month sum aggregation')
    parser.add_argument('-mapgrid', type=str, help='Map grid list')
    parser.add_argument('-radius', type=float, help='Regional fit radius')
    parser.add_argument('-beta', action='store_true', help='Beta only')
    parser.add_argument('-debug', action='store_true', help='Debug mode')
    parser.add_argument('-optim', action='store_true', help='Optimize Weibull')
    parser.add_argument('-plimits', action='store_true', help='Prediction limits')
    parser.add_argument('-weibonly', action='store_true', help='Weibull only')
    parser.add_argument('-subset', action='store_true', help='Annual subset')
    parser.add_argument('-paronly', action='store_true', help='Write parameters only')
    parser.add_argument('-writefits', action='store_true', help='Write fits')
    parser.add_argument('-solve', nargs=8, type=float, help='Solve parameters')
    parser.add_argument('-sqlsolve', nargs=2, type=str, help='SQL solve parameters')
    parser.add_argument('-stochas', type=str, help='Stochastic mode parameters')

    args = parser.parse_args()

    # Initialization and setting defaults
    mpleshome = os.environ['TAOS_HOME']
    debug = False
    map_grid = {'xspan': 0}
    out_base = 'tstat'
    param_prefix = ''
    ext_numvals = -1
    write_fits = False
    ishurricane = False
    analysis_mode = None

    print("\nTAOS(tm) Hazard Frequency Analysis System")
    BuildData = "Build Information"
    CopyrightData = "© 2025 Company Name"
    print(BuildData)
    print(CopyrightData)
    print("\n")

    # Command-line argument parsing
    if args.fitext:
        fit_file = args.fitext
        analysis_mode = "MODE_EXTERNAL"
    if args.table:
        sql_tab_name = args.table
        fit_results_file = args.table
    if args.thres:
        master_threshold = args.thres
    if args.ncscale:
        ncscale = args.ncscale
    if args.pre:
        param_prefix = args.pre
    if args.numvals:
        ext_numvals = args.numvals
    if args.ncfiles:
        nc_file_list = args.ncfiles
    if args.ncvar:
        nc_varname = args.ncvar
    if args.points:
        point_file_list = args.points
        analysis_mode = "MODE_INTERNAL"
        do_points = True
    if args.monthspi:
        spi_refyy, spi_refmm, spi_yy, spi_mm = args.monthspi
        spi_mode = True
        rp_ag_type = "RP_AGTYPE_SUM"
        rp_ag_mode = "RP_AGMODE_MONTHLY"
    if args.spi:
        spi_mode = True
        spi_query_value = args.spi
    if args.mfv:
        max_fit_vals = args.mfv
    if args.march:
        march_size = args.march
    if args.eval:
        eval_map_file = args.eval
        eval_map = True
        analysis_mode = "MODE_INTERNAL"
    if args.rpmaps:
        rpmap_list = args.rpmaps
        analysis_mode = "MODE_INTERNAL"
        do_maps = True
    if args.annualsum:
        rp_ag_type = "RP_AGTYPE_SUM"
        rp_ag_mode = "RP_AGMODE_ANNUAL"
    if args.annualmax:
        rp_ag_type = "RP_AGTYPE_MAX"
        rp_ag_mode = "RP_AGMODE_ANNUAL"
    if args.monthmax:
        rp_ag_type = "RP_AGTYPE_MAX"
        target_month = args.monthmax
        rp_ag_mode = "RP_AGMODE_MONTHLY"
    if args.monthsum:
        rp_ag_type = "RP_AGTYPE_SUM"
        target_month = args.monthsum
        rp_ag_mode = "RP_AGMODE_MONTHLY"
    if args.mapgrid:
        map_grid_list = args.mapgrid
        map_grid = read_grid_spec(map_grid_list)
        user_grid = True
    if args.radius:
        regfit_radius = args.radius
        regfit = True
    if args.beta:
        betaonly = True
    if args.debug:
        debug = True
    if args.optim:
        optimize_weib = True
    if args.plimits:
        map_plimits = True
    if args.weibonly:
        weibull_only = True
    if args.subset:
        annual_subset = True
    if args.paronly:
        write_params = True
    if args.writefits:
        write_fits = True
    if args.solve:
        fit_file, tfield, weib_fit_param1, weib_fit_param2, weib_fit_sd_param1, weib_fit_sd_param2, weib_fit_correl, weib_fit_abs_zero_frac = args.solve
        analysis_mode = "MODE_SOLVE"
        weib_fit = {
            'param1': weib_fit_param1,
            'param2': weib_fit_param2,
            'sd_param1': weib_fit_sd_param1,
            'sd_param2': weib_fit_sd_param2,
            'correl': weib_fit_correl,
            'abs_zero_frac': weib_fit_abs_zero_frac,
            'valid': True
        }
    if args.sqlsolve:
        fit_file, tvarnam = args.sqlsolve
        analysis_mode = "MODE_SOLVE_SQL"
    if args.stochas:
        rpmap_list = args.stochas
        analysis_mode = "MODE_STOCHAST"

    # Perform the analysis
    if analysis_mode == "MODE_EXTERNAL":
        analyze_external_data()
    elif analysis_mode == "MODE_INTERNAL":
        if do_maps:
            analyze_area()
        if eval_map:
            evaluate_map()
        if do_points:
            analyze_points()
    elif analysis_mode == "MODE_SOLVE":
        solve_external_data()
    elif analysis_mode == "MODE_SOLVE_SQL":
        solve_external_data_sql()
    elif analysis_mode == "MODE_STOCHAST":
        create_annmax_nc()
        analyze_anmax()
    else:
        print("ERROR: No analysis specified!")

if __name__ == "__main__":
    main()
    
    

def num_records(filename):
    with open(filename, 'r') as file:
        return sum(1 for _ in file)

def qsort(arr):
    arr.sort()


def weibd(param1, param2, value):
    # Compute the Weibull distribution value
    return weibull_min.cdf(value, param1, scale=param2)

def probnorm(value):
    # Compute the normal probability
    return norm.cdf(value)


def analyze_external_data(fit_file, fit_results_file, param_prefix, write_fits, debug, ext_numvals, spi_query_value, spi_mode, ishurricane, weib_fit, lognorm_fit, write_params, weibull_only):
    print(f"Analyzing external data file {fit_file}")

    if not os.path.exists(fit_file):
        print(f"EPIC FAIL: {fit_file} does not exist.")
        return

    numvals = num_records(fit_file)
    if ext_numvals > numvals:
        if debug:
            print(f"{numvals} in file, {ext_numvals} total values")
        numvals = ext_numvals

    rvals = np.zeros(numvals)
    svals = np.zeros(numvals)

    with open(fit_file, 'r') as file:
        i = 0
        for line in file:
            try:
                rvals[i] = float(line.strip())
                i += 1
            except ValueError:
                continue

    if i < numvals:
        rvals[i:numvals] = 0

    if debug:
        print(f"Got {numvals} valid values from external file.")

    svals[:] = np.sort(rvals)

    with open('empirical.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['p', 'rpd', 'val'])
        for i in range(1, numvals):
            r = (i * 1.0) / numvals
            rp = 1.0 / (1.0 - r)
            writer.writerow([r, rp, svals[i]])

    print("  Fitting data . . .")
    fit_data(rvals, numvals)

    if write_params:
        params_filename = f"{fit_results_file}_params.csv"
        with open(params_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Prefix', 'Type', 'Param1', 'Param2', 'SD_Param1', 'SD_Param2', 'Correl', 'Abs_Zero_Frac', 'Chisq'])
            writer.writerow([param_prefix, 'Weibull', weib_fit['param1'], weib_fit['param2'], weib_fit['sd_param1'], weib_fit['sd_param2'], weib_fit['correl'], weib_fit['abs_zero_frac'], weib_fit['chisq']])
            if not weibull_only:
                writer.writerow([param_prefix, 'LogNormal', lognorm_fit['param1'], lognorm_fit['param2'], lognorm_fit['sd_param1'], lognorm_fit['sd_param2'], lognorm_fit['correl'], lognorm_fit['abs_zero_frac'], lognorm_fit['chisq']])
        if not write_fits:
            print("Done.")
            return

    if spi_mode:
        rp = weibd(weib_fit['param1'], weib_fit['param2'], spi_query_value)
        r = probnorm(rp)
        print(f"SPI: {spi_query_value}, {rp}, {r}")

    else:
        print("  Generating analysis file . . .")
        analysis_filename = f"{fit_results_file}_analysis.csv"
        with open(analysis_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['rp', 'rpval', 'mle', 'p01', 'p05', 'p10', 'p25', 'p33', 'p50', 'p67', 'p75', 'p90', 'p95', 'p99'])
            stfac = 1.0
            i = 1
            r = -0.5

            with open('point_multianalysis.csv', 'w', newline='') as pointfile:
                point_writer = csv.writer(pointfile)
                point_writer.writerow(['rp', 'weib', 'lnorm'])
                while True:
                    delta_i = 1
                    if i > 200:
                        delta_i = 2
                    if i > 1000:
                        delta_i = 5
                    if i > 10000:
                        delta_i = 10
                    i += delta_i
                    if i > numvals * 5:
                        break
                    rp = 1.0 - (1.0 / (i / stfac))
                    generate_weibull_std_plims(rp, mle, p10, p25, p33, p50, p67, p75, p90, p01, p05, p95, p99)
                    flag, r = get_weib_val(rp, r, flag)
                    if not flag:
                        p01 = p05 = p10 = p25 = p33 = p50 = p67 = p75 = p90 = p95 = p99 = mle = r
                    writer.writerow([i / stfac, rp, mle, p01, p05, p10, p25, p33, p50, p67, p75, p90, p95, p99])
                    wb, lno = get_all_rp_val(rp, wb, lno, ev1l, ev1u, flag)
                    point_writer.writerow([i / stfac, wb, lno])

    if ishurricane:
        tval = 64.0 / 1.94
        mlepval = weibd(weib_fit['param1'], weib_fit['param2'], tval) * (1.0 - weib_fit['abs_zero_frac']) + weib_fit['abs_zero_frac']
        print(f"Hurricane Wind RP: {int(1.0 / (1.0 - mlepval))}")
        tval = 34.0 / 1.94
        mlepval = weibd(weib_fit['param1'], weib_fit['param2'], tval) * (1.0 - weib_fit['abs_zero_frac']) + weib_fit['abs_zero_frac']
        print(f"TropStorm Wind RP: {int(1.0 / (1.0 - mlepval))}")

        icon = 'thunderstorm_green.png'
        rp = 0.95
        generate_weibull_std_plims(rp, mle, p10, p25, p33, p50, p67, p75, p90, p01, p05, p95, p99)
        if mle * 1.94 >= 34:
            icon = 'Tropical_Storm_north.png'
        if mle * 1.94 >= 50:
            icon = 'Strong_Tropical_Storm_north.png'
        if mle * 1.94 >= 64:
            icon = 'Saffir-Simpson_Category_1.png'
        if mle * 1.94 >= 83:
            icon = 'Saffir-Simpson_Category_2.png'
        if mle * 1.94 >= 96:
            icon = 'Saffir-Simpson_Category_3.png'
        if mle * 1.94 >= 113:
            icon = 'Saffir-Simpson_Category_4.png'
        if mle * 1.94 >= 137:
            icon = 'Saffir-Simpson_Category_5.png'

        print(f"Five percent wind: {int(mle * 1.94)}")
        print(f"Icon: {icon}")

# Example usage
fit_file = 'example_fit_file.txt'
fit_results_file = 'example_fit_results_file'
param_prefix = 'example_prefix'
write_fits = False
debug = False
ext_numvals = 0
spi_query_value = 0.0
spi_mode = False
ishurricane = False
weibull_only = False
write_params = False
lognorm_fit = {
    'param1': 0.0,
    'param2': 0.0,
    'sd_param1': 0.0,
    'sd_param2': 0.0,
    'correl': 0.0,
    'abs_zero_frac': 0.0,
    'chisq': 0.0
}
weib_fit = {
    'param1': 0.0,
    'param2': 0.0,
    'sd_param1': 0.0,
    'sd_param2': 0.0,
    'correl': 0.0,
    'abs_zero_frac': 0.0,
    'chisq': 0.0
}

analyze_external_data(fit_file, fit_results_file, param_prefix, write_fits, debug, ext_numvals, spi_query_value, spi_mode, ishurricane, weib_fit, lognorm_fit, write_params, weibull_only)



def analyze_points(point_file_list, nc_files, nc_varname, num_ncfiles, num_points, rp_ag_mode, rp_ag_type, end_year, base_year, ext_numvals, debug, write_params, weib_fit, lognorm_fit, weibull_only, write_fits, annual_subset, target_month, regfit, regfit_radius, nc_grid, pt_lat, pt_lon, point_names):
    print(f"Analyzing point file {point_file_list}")
    read_point_list()
    load_ncfile_list()
    get_date_range()
    
    if rp_ag_mode == 'RP_AGMODE_NONE':
        print('  No aggregation ')
    elif rp_ag_mode == 'RP_AGMODE_ANNUAL':
        print('  Annual Aggregation ')
    elif rp_ag_mode == 'RP_AGMODE_MONTHLY':
        print('  Monthly Aggregation ')

    if rp_ag_type == 'RP_AGTYPE_SUM':
        print('  Summing Values')
    elif rp_ag_type == 'RP_AGTYPE_MAX':
        print('  Storing Max Value')

    num_rvals = num_ncfiles
    if rp_ag_mode == 'RP_AGMODE_MONTHLY':
        num_rvals = end_year - base_year
    if rp_ag_mode == 'RP_AGMODE_ANNUAL':
        num_rvals = end_year - base_year

    pt_vals = np.zeros((num_points, num_rvals))
    print(f'rvals: {num_ncfiles}')
    numvals = num_rvals

    if ext_numvals > numvals:
        if debug:
            print(f"{numvals} in file, {ext_numvals} total values")
        numvals = ext_numvals

    print('  Processing netcdf files . . . ')
    for i in range(num_ncfiles):
        if debug:
            print(f'  Processing {nc_files[i]} {nc_varname}')
        errorflag = False
        get_nc_vals(nc_files[i], nc_varname, errorflag)

        for j in range(num_points):
            x, y, ongrid = ll_to_gridxy(nc_grid, pt_lat[j], pt_lon[j], x, y, ongrid)

            if rp_ag_mode == 'RP_AGMODE_NONE':
                idx = i
            elif rp_ag_mode == 'RP_AGMODE_ANNUAL':
                yy, mm, dd, rhr = caldat(nc_jday[i], yy, mm, dd, rhr)
                idx = yy - base_year
            elif rp_ag_mode == 'RP_AGMODE_MONTHLY':
                yy, mm, dd, rhr = caldat(nc_jday[i], yy, mm, dd, rhr)
                idx = yy - base_year
                if mm != target_month:
                    continue

            rval = 0.0
            if ongrid:
                rval = 0.0
                xp = x
                yp = y
                if ongrid:
                    if regfit:
                        rval = 0.0
                        for xz in range(xp - regfit_radius, xp + regfit_radius + 1):
                            if xz < 1 or xz > nc_grid['xspan']:
                                continue
                            for yz in range(yp - regfit_radius, yp + regfit_radius + 1):
                                if yz < 1 or yz > nc_grid['yspan']:
                                    continue
                                if nc_vals[xz, yz] > rval:
                                    rval = nc_vals[xz, yz]
                    else:
                        rval = nc_vals[xp, yp]

            if rp_ag_type == 'RP_AGTYPE_NONE':
                pt_vals[j, idx] = rval
            elif rp_ag_type == 'RP_AGTYPE_SUM':
                pt_vals[j, idx] += rval
            elif rp_ag_type == 'RP_AGTYPE_MAX':
                if pt_vals[j, idx] < rval:
                    pt_vals[j, idx] = rval

    print('  Processing complete.')

    svals = np.zeros(num_rvals)
    rvals = np.zeros(num_rvals)
    print('  Fitting points', num_rvals)
    if write_params:
        with open('point_fit_params.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

    for j in range(num_points):
        fit_results_file = point_names[j]
        rvals[:] = 0
        svals[:] = 0
        numanvals = 0
        for i in range(num_rvals):
            if not annual_subset or pt_vals[j, i] > 0:
                numanvals += 1
                rvals[numanvals - 1] = pt_vals[j, i]

        if numanvals < ext_numvals:
            numanvals = ext_numvals

        svals[:] = np.sort(rvals[:numanvals])

        fit_data(rvals, numanvals)

        if write_params:
            with open('point_fit_params.csv', 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([point_names[j], 'Weibull', weib_fit['param1'], weib_fit['param2'], weib_fit['sd_param1'], weib_fit['sd_param2'], weib_fit['correl'], weib_fit['chisq'], weib_fit['abs_zero_frac']])
                if not weibull_only:
                    writer.writerow([point_names[j], 'LogNormal', lognorm_fit['param1'], lognorm_fit['param2'], lognorm_fit['sd_param1'], lognorm_fit['sd_param2'], lognorm_fit['correl'], lognorm_fit['chisq'], lognorm_fit['abs_zero_frac']])
        else:
            fname = f"{fit_results_file}_empirical.csv"
            with open(fname, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['p', 'rpd', 'val'])
                for i in range(1, numvals):
                    r = (i * 1.0) / numvals
                    rp = 1.0 / (1.0 - r)
                    writer.writerow([r, rp, svals[i]])

            print(f'  Generating analysis file . . . {point_names[j]}')

            fname = f"{fit_results_file}_analysis.csv"
            with open(fname, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['rp', 'rpval', 'mle', 'p10', 'p25', 'p33', 'p50', 'p67', 'p75', 'p90'])
                for i in range(20, numanvals * 50):
                    rp = 1.0 - (1.0 / (i / 10.0))
                    generate_weibull_std_plims(rp, mle, p10, p25, p33, p50, p67, p75, p90, p01, p05, p95, p99)
                    writer.writerow([i / 10.0, rp, mle, p10, p25, p33, p50, p67, p75, p90])

    if write_params:
        print('Point processing complete.')
    write_fits = False

# Example usage
point_file_list = 'example_point_file_list.txt'
nc_files = ['example_nc_file1.nc', 'example_nc_file2.nc']
nc_varname = 'example_varname'
num_ncfiles = len(nc_files)
num_points = 10
rp_ag_mode = 'RP_AGMODE_ANNUAL'
rp_ag_type = 'RP_AGTYPE_SUM'
end_year = 2025
base_year = 2020
ext_numvals = 0
debug = False
write_params = False
weib_fit = {
    'param1': 0.0,
    'param2': 0.0,
    'sd_param1': 0.0,
    'sd_param2': 0.0,
    'correl': 0.0,
    'chisq': 0.0,
    'abs_zero_frac': 0.0
}
lognorm_fit = {
    'param1': 0.0,
    'param2': 0.0,
    'sd_param1': 0.0,
    'sd_param2': 0.0,
    'correl': 0.0,
    'abs_zero_frac': 0.0,
    'chisq': 0.0
}
weibull_only = False
write_fits = False
annual_subset = False
target_month = 1
regfit = False
regfit_radius = 0
nc_grid = {'xspan': 100, 'yspan': 100}
pt_lat = np.zeros(num_points)
pt_lon = np.zeros(num_points)
point_names = [f'point_{i}' for i in range(num_points)]

analyze_points(point_file_list, nc_files, nc_varname, num_ncfiles, num_points, rp_ag_mode, rp_ag_type, end_year, base_year, ext_numvals, debug, write_params, weib_fit, lognorm_fit, weibull_only, write_fits, annual_subset, target_month, regfit, regfit_radius, nc_grid, pt_lat, pt_lon, point_names)



def initialize_nc_anmax(out_base, map_grid, nyears):
    # Initialize the NetCDF file for annual maxima
    ncfname = f"{out_base}_annmax.nc"
    ncid = Dataset(ncfname, 'w', format='NETCDF4')
    lat_dim = ncid.createDimension('lat', map_grid['yspan'])
    lon_dim = ncid.createDimension('lon', map_grid['xspan'])
    year_dim = ncid.createDimension('year', nyears)

    lats = ncid.createVariable('latitude', 'f4', ('lat',))
    lons = ncid.createVariable('longitude', 'f4', ('lon',))
    annmax = ncid.createVariable('annmax', 'f4', ('year', 'lat', 'lon',))

    # Fill lats and lons with dummy data
    lats[:] = np.linspace(-90, 90, map_grid['yspan'])
    lons[:] = np.linspace(-180, 180, map_grid['xspan'])

    return ncid, annmax

def nfcheck(status):
    # Placeholder for checking NetCDF status
    if status != 0:
        raise Exception("NetCDF error")

def create_annmax_nc(out_base, map_grid, num_ncfiles, nc_files, nc_varname, valid_nc, nc_jday, base_year, end_year):
    print("\n*** Stochastic/Raw Annual Big Data Mode ***\n")
    load_ncfile_list()
    get_date_range()
    nyears = end_year - base_year

    print("\n")
    ncfname = f"{out_base}_annmax.nc"
    if os.path.exists(ncfname):
        print(f"{ncfname} exists, cowardly refusing to over-write it!\n")
        return

    yrncid, annmax_var = initialize_nc_anmax(out_base, map_grid, nyears)
    anmax = np.zeros((map_grid['xspan'], map_grid['yspan']))
    start = [1, 1, 0]

    for yr in range(base_year + 1, end_year + 1):
        stime = time.time()
        nevents = 0
        print(f"   Processing {yr}")
        anmax.fill(0.0)
        for i in range(num_ncfiles):
            if not valid_nc[i]:
                continue
            yy, mm, dd, rhr = caldat(nc_jday[i])
            if yy != yr:
                continue
            nc_vals, errorflag = get_nc_vals(nc_files[i], nc_varname)
            if errorflag:
                print(f"Rejecting {nc_files[i]}")
                continue
            gequiv = grids_equiv(map_grid, nc_grid)

            for x in range(map_grid['xspan']):
                for y in range(map_grid['yspan']):
                    if gequiv:
                        rval = nc_vals[x, y]
                        if rval >= 9e30:
                            rval = 0.0
                    else:
                        lat, lon = gridxy_to_ll(map_grid, x, y)
                        xp, yp, ongrid = ll_to_gridxy(nc_grid, lat, lon)
                        rval = 0.0
                        if ongrid:
                            rval = nc_vals[xp, yp]
                        if rval >= 9e30:
                            rval = 0.0
                    if rval > anmax[x, y]:
                        anmax[x, y] = rval

            start[2] = yr - base_year
            nfcheck(annmax_var[start[2]:start[2]+1, :, :] == anmax)
            nevents += 1
        etime = time.time() - stime
        print(f"{yr} had {nevents} max {np.max(anmax)} time {etime}")
    
    yrncid.close()
    print("Completed processing.")

# Example usage
out_base = 'example_output'
map_grid = {'xspan': 100, 'yspan': 100}
num_ncfiles = 5
nc_files = [f'example_nc_file_{i}.nc' for i in range(num_ncfiles)]
nc_varname = 'example_varname'
valid_nc = [True] * num_ncfiles
nc_jday = [2459215 + i for i in range(num_ncfiles)]  # Example Julian dates
base_year = 2020
end_year = 2025

create_annmax_nc(out_base, map_grid, num_ncfiles, nc_files, nc_varname, valid_nc, nc_jday, base_year, end_year)



def initialize_nc_anmax(out_base, map_grid, base_year, end_year):
    """
    Initialize the NetCDF file for annual maxima.
    
    Parameters:
    out_base (str): Base name for the output file.
    map_grid (dict): Dictionary containing grid specifications.
    base_year (int): Start year of the data.
    end_year (int): End year of the data.
    
    Returns:
    Dataset: NetCDF dataset object.
    Variable: NetCDF variable for annual maxima.
    """
    print('Initializing netcdf format Annual Maxima output file . . .')
    nyears = end_year - base_year + 1

    ncfname = f"{out_base}_annmax.nc"
    ncid = Dataset(ncfname, 'w', format='NETCDF4')

    lat_dim = ncid.createDimension('latitude', map_grid['yspan'])
    lon_dim = ncid.createDimension('longitude', map_grid['xspan'])
    year_dim = ncid.createDimension('year', nyears)

    lat_var = ncid.createVariable('latitude', 'f4', ('latitude',))
    lon_var = ncid.createVariable('longitude', 'f4', ('longitude',))
    annmax_var = ncid.createVariable('annmax', 'f4', ('year', 'latitude', 'longitude'))

    ncid.setncattr_string('units', 'degrees')
    lat_var.units = 'degrees_north'
    lon_var.units = 'degrees_east'

    lats = np.zeros(map_grid['yspan'])
    lons = np.zeros(map_grid['xspan'])

    if map_grid['reversed']:
        print('Writing reversed format')
        for i in range(map_grid['yspan']):
            lats[i] = map_grid['ymax'] - (i + 1) * map_grid['yres']
    else:
        for i in range(map_grid['yspan']):
            lats[i] = map_grid['ymin'] + (i + 1) * map_grid['yres']

    for i in range(map_grid['xspan']):
        lons[i] = map_grid['xmin'] + (i + 1) * map_grid['xres']

    lat_var[:] = lats
    lon_var[:] = lons

    ncid.sync()
    return ncid, annmax_var

# Example usage
out_base = 'example_output'
map_grid = {'xspan': 100, 'yspan': 100, 'xmin': 0.0, 'xmax': 100.0, 'ymin': 0.0, 'ymax': 100.0, 'xres': 1.0, 'yres': 1.0, 'reversed': False}
base_year = 2020
end_year = 2025

ncid, annmax_var = initialize_nc_anmax(out_base, map_grid, base_year, end_year)
print("NetCDF file initialized.")
ncid.close()



def analyze_anmax(out_base, map_grid, base_year, end_year, num_rpmaps, return_periods):
    """
    Analyze annual maxima and generate return period maps.
    
    Parameters:
    out_base (str): Base name for the output file.
    map_grid (dict): Dictionary containing grid specifications.
    base_year (int): Start year of the data.
    end_year (int): End year of the data.
    num_rpmaps (int): Number of return period maps.
    return_periods (list): List of return periods.
    """
    read_rpmap_list()
    initialize_rpmaps()
    
    ncfname = f"{out_base}_annmax.nc"
    ncid = Dataset(ncfname, 'r')
    yr_varid = ncid.variables['annmax']
    nyears = end_year - base_year + 1
    
    maxvals = np.zeros(nyears)
    maxrow = np.zeros((map_grid['xspan'], nyears))
    rpmap = np.zeros((map_grid['xspan'], map_grid['yspan'], num_rpmaps))
    
    print(f"Analyzing annual max file with {nyears} to generate return period maps")
    
    for y in range(map_grid['yspan']):
        stime = time.time()
        for i in range(nyears):
            rowline = yr_varid[i, y, :]
            maxrow[:, i] = rowline
            
        for x in range(map_grid['xspan']):
            maxvals.fill(0)
            for i in range(nyears):
                maxvals[i] = maxrow[x, i]
                if maxvals[i] > 9e30:
                    maxvals[i] = 0
            qsort(maxvals)
            
            for ri in range(num_rpmaps):
                rp = 1.0 - 1.0 / return_periods[ri]
                idx = int(nyears * rp)
                if idx < 1:
                    idx = 1
                if idx >= nyears:
                    idx = nyears - 1
                rpmap[x, y, ri] = maxvals[idx]
        
        etime = time.time() - stime
        print(f"{y + 1} of {map_grid['yspan']} time {etime}")
    
    print("\n      Index   Rtn Pd          RtnFrac               YrIdx   MaxGridVal")
    for ri in range(num_rpmaps):
        rp = 1.0 - 1.0 / return_periods[ri]
        idx = int(nyears * rp)
        if idx < 1:
            idx = 1
        if idx >= nyears:
            idx = nyears - 1
        anmax = rpmap[:, :, ri]
        ncid.variables[f'rp{ri}'][:] = anmax
        print(f"{ri} {return_periods[ri]} {rp} {idx} {np.max(anmax)}")
    
    ncid.close()

# Example usage
out_base = 'example_output'
map_grid = {'xspan': 100, 'yspan': 100}
base_year = 2020
end_year = 2025
num_rpmaps = 5
return_periods = [2, 5, 10, 25, 50]

analyze_anmax(out_base, map_grid, base_year, end_year, num_rpmaps, return_periods)



def initialize_rpmaps(out_base, map_grid, base_year, end_year, num_rpmaps, rp_names):
    """
    Initialize the NetCDF file for return period maps.
    
    Parameters:
    out_base (str): Base name for the output file.
    map_grid (dict): Dictionary containing grid specifications.
    base_year (int): Start year of the data.
    end_year (int): End year of the data.
    num_rpmaps (int): Number of return period maps.
    rp_names (list): List of return period map names.
    
    Returns:
    Dataset: NetCDF dataset object.
    list: NetCDF variables for return period maps.
    """
    print('Initializing netcdf format Return Period output file . . .')
    nyears = end_year - base_year + 1

    ncfname = f"{out_base}_rpmaps.nc"
    ncid = Dataset(ncfname, 'w', format='NETCDF4')

    lat_dim = ncid.createDimension('latitude', map_grid['yspan'])
    lon_dim = ncid.createDimension('longitude', map_grid['xspan'])

    lat_var = ncid.createVariable('latitude', 'f4', ('latitude',))
    lon_var = ncid.createVariable('longitude', 'f4', ('longitude',))

    lat_var.units = 'degrees_north'
    lon_var.units = 'degrees_east'

    lats = np.zeros(map_grid['yspan'])
    lons = np.zeros(map_grid['xspan'])

    if map_grid['reversed']:
        print('Writing reversed format')
        for i in range(map_grid['yspan']):
            lats[i] = map_grid['ymax'] - (i + 1) * map_grid['yres']
    else:
        for i in range(map_grid['yspan']):
            lats[i] = map_grid['ymin'] + (i + 1) * map_grid['yres']

    for i in range(map_grid['xspan']):
        lons[i] = map_grid['xmin'] + (i + 1) * map_grid['xres']

    lat_var[:] = lats
    lon_var[:] = lons

    rp_vars = []
    for rp_name in rp_names:
        rp_var = ncid.createVariable(
            rp_name, 'f4', ('latitude', 'longitude'),
            zlib=True, shuffle=True, complevel=5, endian='native'
        )
        rp_vars.append(rp_var)

    ncid.sync()
    return ncid, rp_vars

# Example usage
out_base = 'example_output'
map_grid = {'xspan': 100, 'yspan': 100, 'xmin': 0.0, 'xmax': 100.0, 'ymin': 0.0, 'ymax': 100.0, 'xres': 1.0, 'yres': 1.0, 'reversed': False}
base_year = 2020
end_year = 2025
num_rpmaps = 5
rp_names = [f'rp_map_{i}' for i in range(num_rpmaps)]

ncid, rp_vars = initialize_rpmaps(out_base, map_grid, base_year, end_year, num_rpmaps, rp_names)
print("NetCDF file for return period maps initialized.")
ncid.close()


def analyze_area(out_base, map_grid, nc_files, nc_varname, valid_nc, nc_jday, base_year, end_year, num_rpmaps, return_periods, rp_names, rp_units, spi_mode=False, spi_refyy=None, spi_refmm=None, spi_yy=None, spi_mm=None, spi_query_value=None, regfit=False, regfit_radius=0, map_plimits=False):
    """
    Analyze area for return period maps.
    
    Parameters:
    out_base (str): Base name for the output file.
    map_grid (dict): Dictionary containing grid specifications.
    nc_files (list): List of netCDF files.
    nc_varname (str): Variable name in netCDF files.
    valid_nc (list): List of valid netCDF files.
    nc_jday (list): List of Julian days corresponding to netCDF files.
    base_year (int): Start year of the data.
    end_year (int): End year of the data.
    num_rpmaps (int): Number of return period maps.
    return_periods (list): List of return periods.
    rp_names (list): List of return period map names.
    rp_units (list): List of return period map units.
    spi_mode (bool): Whether to run in SPI mode.
    spi_refyy (int): SPI reference year.
    spi_refmm (int): SPI reference month.
    spi_yy (int): SPI year.
    spi_mm (int): SPI month.
    spi_query_value (float): SPI query value.
    regfit (bool): Whether to use regional fit.
    regfit_radius (int): Radius for regional fit.
    map_plimits (bool): Whether to compute prediction limits.
    """
    load_ncfile_list()
    get_date_range()

    print("\nAnalyzing area for return period maps . . .")
    if map_grid['xspan'] == 0:
        print("  Using netcdf grid specifications for mapping")
        map_grid = nc_grid
    else:
        print("  Using user specified grid.")

    read_rpmap_list()
    if num_rpmaps == 0:
        return
    
    num_rvals = len(nc_files)
    if rp_ag_mode == 'RP_AGMODE_MONTHLY':
        num_rvals = (end_year - base_year) * 12
    if rp_ag_mode == 'RP_AGMODE_ANNUAL':
        num_rvals = end_year - base_year

    print(f"Allocating {map_grid['xspan']} {map_grid['yspan']} {num_rvals}")
    cell_vals = np.zeros(num_rvals)
    map_fit = np.zeros((map_grid['xspan'], map_grid['yspan']), dtype=object)
    rawdata = np.zeros((map_grid['xspan'], map_grid['yspan'], num_rvals))

    print("  Collecting data . . .")
    for i in range(len(nc_files)):
        if not valid_nc[i]:
            continue
        nc_vals, errorflag = get_nc_vals(nc_files[i], nc_varname)
        if errorflag:
            print(f"Rejecting {nc_files[i]}")
            continue

        gequiv = grids_equiv(map_grid, nc_grid)
        for x in range(map_grid['xspan']):
            for y in range(map_grid['yspan']):
                if gequiv:
                    rval = nc_vals[x, y]
                else:
                    lat, lon = gridxy_to_ll(map_grid, x, y)
                    xp, yp, ongrid = ll_to_gridxy(nc_grid, lat, lon)
                    rval = 0.0
                    if ongrid:
                        if regfit:
                            rval = 0.0
                            for xz in range(xp - regfit_radius, xp + regfit_radius + 1):
                                if xz < 1 or xz > nc_grid['xspan']:
                                    continue
                                for yz in range(yp - regfit_radius, yp + regfit_radius + 1):
                                    if yz < 1 or yz > nc_grid['yspan']:
                                        continue
                                    if nc_vals[xz, yz] > rval:
                                        rval = nc_vals[xz, yz]
                        else:
                            rval = nc_vals[xp, yp]
                
                if rp_ag_mode == 'RP_AGMODE_NONE':
                    idx = i
                elif rp_ag_mode == 'RP_AGMODE_ANNUAL':
                    yy, mm, dd, rhr = caldat(nc_jday[i])
                    idx = yy - base_year
                elif rp_ag_mode == 'RP_AGMODE_MONTHLY':
                    yy, mm, dd, rhr = caldat(nc_jday[i])
                    idx = (yy - base_year - 1) * 12 + mm

                if rp_ag_type == 'RP_AGTYPE_NONE':
                    rawdata[x, y, idx] = rval
                elif rp_ag_type == 'RP_AGTYPE_SUM':
                    rawdata[x, y, idx] += rval
                elif rp_ag_type == 'RP_AGTYPE_MAX':
                    if rawdata[x, y, idx] < rval:
                        rawdata[x, y, idx] = rval

    if spi_mode:
        rawdata *= 3937.0

    print("  Generating fits . . .")
    num_done = 0
    num_tot = map_grid['xspan'] * map_grid['yspan']

    ref_pd = num_rvals
    if spi_mode:
        ref_pd = (spi_refyy - base_year - 1) * 12 + spi_refmm

    print(f"spi ref: {num_rvals} {ref_pd}")
    for x in range(map_grid['xspan']):
        for y in range(map_grid['yspan']):
            num_done += 1
            cell_vals.fill(0.0)
            for i in range(ref_pd):
                cell_vals[i] = rawdata[x, y, i]
            map_fit[x, y] = {"valid": False}
            if np.max(cell_vals) != 0:
                fit_data(cell_vals, ref_pd)
                map_fit[x, y] = weib_fit

    rp_outvals = np.zeros((map_grid['xspan'], map_grid['yspan']))
    if map_plimits:
        weibvals = np.zeros((map_grid['xspan'], map_grid['yspan'], 8))
    initialize_nc_outfile()

    for i in range(num_rpmaps):
        rp_outvals.fill(0.0)
        rpval = 1.0 - 1.0 / return_periods[i]
        print(f"  Generating {rp_names[i]} {rpval}")
        nc_varunits = rp_units[i]
        for x in range(map_grid['xspan']):
            for y in range(map_grid['yspan']):
                if map_fit[x, y]["valid"]:
                    weib_fit = map_fit[x, y]
                    mle = generate_weibull_mle(rpval)
                    rp_outvals[x, y] = mle
                    if map_plimits:
                        fitvals = generate_weibull_std_plims(rpval)
                        for j in range(8):
                            weibvals[x, y, j] = fitvals[j]
                else:
                    rp_outvals[x, y] = 0.0
                    if map_plimits:
                        weibvals[x, y].fill(0.0)

        if map_plimits:
            for j in range(8):
                for x in range(map_grid['xspan']):
                    for y in range(map_grid['yspan']):
                        rp_outvals[x, y] = weibvals[x, y, j]
                limname = f"{rp_names[i]}_{pnames[j]}"
                write_nc_grid(limname, rp_units[i])
        else:
            write_nc_grid(rp_names[i], rp_units[i])

    if rp_ag_mode == 'RP_AGMODE_ANNUAL':
        rp_outvals.fill(0.0)
        for x in range(map_grid['xspan']):
            for y in range(map_grid['yspan']):
                for i in range(num_rvals):
                    rp_outvals[x, y] += rawdata[x, y, i]
        rp_outvals /= num_rvals
        write_nc_grid('average', 'per year')

    if spi_mode:
        idx = (spi_yy - base_year - 1) * 12 + spi_mm
        print(f"Generating SPI for {spi_yy} {spi_mm} {idx}")
        for x in range(map_grid['xspan']):
            for y in range(map_grid['yspan']):
                weib_fit = map_fit[x, y]
                spi_query_value = rawdata[x, y, idx]
                rp = weibd(weib_fit['param1'], weib_fit['param2'], spi_query_value)
                r = probnorm(rp)
                rp_outvals[x, y] = r
        write_nc_grid('spi', 'sigma')

    del rp_outvals

# Example usage
out_base = 'example_output'
map_grid = {'xspan': 100, 'yspan': 100}
nc_files = ['example_nc_file_1.nc', 'example_nc_file_2.nc']
nc_varname = 'example_varname'
valid_nc = [True, True]
nc_jday = [2459215, 2459216]  # Example Julian dates
base_year = 2020
end_year = 2025
num_rpmaps = 5
return_periods = [2, 5, 10, 25, 50]
rp_names = [f'rp_map_{i}' for i in range(num_rpmaps)]
rp_units = ['unit'] * num_rpmaps

analyze_area(out_base, map_grid, nc_files, nc_varname, valid_nc, nc_jday, base_year, end_year, num_rpmaps, return_periods, rp_names, rp_units)



def num_records(filename):
    with open(filename, 'r') as file:
        return sum(1 for _ in file)

def parse_csv_line(line, index):
    return line.split(',')[index].strip()

def read_point_list(point_file_list, debug=False):
    """
    Read point list from the specified file.
    
    Parameters:
    point_file_list (str): Path to the point file list.
    debug (bool): Debug mode.
    
    Returns:
    tuple: (point_names, pt_lat, pt_lon, num_points)
    """
    if not os.path.exists(point_file_list):
        return [], [], [], 0

    maxpts = num_records(point_file_list)
    point_names = [None] * maxpts
    pt_lat = np.zeros(maxpts)
    pt_lon = np.zeros(maxpts)

    num_points = 0
    with open(point_file_list, 'r') as file:
        for line in file:
            if line.startswith('#'):
                continue
            num_points += 1
            point_names[num_points - 1] = parse_csv_line(line, 0)
            try:
                pt_lat[num_points - 1] = float(parse_csv_line(line, 1))
                pt_lon[num_points - 1] = float(parse_csv_line(line, 2))
            except ValueError:
                if debug:
                    print(f"Error parsing line: {line.strip()}")
                num_points -= 1

    print(f"Read {num_points} analysis points.")
    return point_names[:num_points], pt_lat[:num_points], pt_lon[:num_points], num_points

def read_rpmap_list(rpmap_list, debug=False):
    """
    Read return period map list from the specified file.
    
    Parameters:
    rpmap_list (str): Path to the return period map list.
    debug (bool): Debug mode.
    
    Returns:
    tuple: (rp_names, return_periods, rp_units, num_rpmaps)
    """
    if not os.path.exists(rpmap_list):
        return [], [], [], 0

    maxpts = num_records(rpmap_list)
    rp_names = [None] * maxpts
    return_periods = np.zeros(maxpts)
    rp_units = [None] * maxpts

    num_rpmaps = 0
    with open(rpmap_list, 'r') as file:
        for line in file:
            if line.startswith('#'):
                continue
            num_rpmaps += 1
            return_periods[num_rpmaps - 1] = int(parse_csv_line(line, 0))
            rp_names[num_rpmaps - 1] = parse_csv_line(line, 1)
            rp_units[num_rpmaps - 1] = parse_csv_line(line, 2)

            if debug:
                print(f"{num_rpmaps} {rp_names[num_rpmaps - 1]} {return_periods[num_rpmaps - 1]} {rp_units[num_rpmaps - 1]}")

    print(f"Got {num_rpmaps} return period map requests.")
    return rp_names[:num_rpmaps], return_periods[:num_rpmaps], rp_units[:num_rpmaps], num_rpmaps

def get_date_range(nc_jday, num_ncfiles):
    """
    Determine the date range from the netCDF Julian days.
    
    Parameters:
    nc_jday (list): List of Julian days.
    num_ncfiles (int): Number of netCDF files.
    
    Returns:
    tuple: (base_year, end_year)
    """
    minjd = min(nc_jday[:num_ncfiles])
    maxjd = max(nc_jday[:num_ncfiles])

    base_year = caldat(minjd)[0]
    end_year = caldat(maxjd)[0]

    print(f"Data range is from {base_year} to {end_year}")
    return base_year - 1, end_year

def read_grid_spec(fname):
    """
    Read grid specification from the specified file.
    
    Parameters:
    fname (str): Path to the grid specification file.
    
    Returns:
    dict: Grid specification.
    """
    if not os.path.exists(fname):
        print(f"EPIC FAIL: {fname} does not exist.")
        return {}

    gspec = {}
    with open(fname, 'r') as file:
        for line in file:
            value = parse_csv_line(line, 0)
            if value == "bounds":
                gspec['ymin'] = float(parse_csv_line(line, 1))
                gspec['xmin'] = float(parse_csv_line(line, 2))
                gspec['ymax'] = float(parse_csv_line(line, 3))
                gspec['xmax'] = float(parse_csv_line(line, 4))
            elif value == "res":
                gspec['xres'] = float(parse_csv_line(line, 1))
                gspec['yres'] = float(parse_csv_line(line, 2))
                gspec['xspan'] = (gspec['xmax'] - gspec['xmin']) / gspec['xres']
                gspec['yspan'] = (gspec['ymax'] - gspec['ymin']) / gspec['yres']
            elif value == "span":
                gspec['xspan'] = int(parse_csv_line(line, 1))
                gspec['yspan'] = int(parse_csv_line(line, 2))
                gspec['xres'] = (gspec['xmax'] - gspec['xmin']) / gspec['xspan']
                gspec['yres'] = (gspec['ymax'] - gspec['ymin']) / gspec['yspan']

    return gspec

def evaluate_map(eval_map_file, nc_varname, rawdata, map_grid, nc_grid, fillvalue):
    """
    Evaluate the map for return periods.
    
    Parameters:
    eval_map_file (str): Path to the evaluation map file.
    nc_varname (str): Variable name in netCDF files.
    rawdata (ndarray): Raw data array.
    map_grid (dict): Map grid specifications.
    nc_grid (dict): NetCDF grid specifications.
    fillvalue (float): Fill value for invalid data.
    
    Returns:
    ndarray: Output values.
    """
    num_vals = rawdata.shape[2]
    cell_vals = np.zeros(num_vals)

    nc_vals, errorflag = get_nc_vals(eval_map_file, nc_varname)
    if errorflag:
        return np.zeros((map_grid['xspan'], map_grid['yspan']))

    rp_outvals = np.zeros((map_grid['xspan'], map_grid['yspan']))

    for xp in range(map_grid['xspan']):
        for yp in range(map_grid['yspan']):
            for i in range(num_vals):
                cell_vals[i] = rawdata[xp, yp, i]
            cell_vals.sort()

            if cell_vals[num_vals - 1] >= 1e20:
                pval = -1
            else:
                rval = nc_vals[xp, yp]
                if rval == fillvalue:
                    pval = -1
                else:
                    pval = np.searchsorted(cell_vals, rval) / num_vals

            rp_outvals[xp, yp] = pval

    return rp_outvals


'''

# Example usage
point_file_list = 'example_point_file_list.txt'
rpmap_list = 'example_rpmap_list.txt'
eval_map_file = 'example_eval_map_file.nc'
nc_varname = 'example_varname'
rawdata = np.random.rand(100, 100, 12)  # Example raw data
map_grid = {'xspan': 100, 'yspan': 100}
nc_grid = {'xspan': 100, 'yspan': 100}
fillvalue = -9999.0

point_names, pt_lat, pt_lon, num_points = read_point_list(point_file_list, debug=True)
rp_names, return_periods, rp_units, num_rpmaps = read_rpmap_list(rpmap_list, debug=True)
base_year, end_year = get_date_range([2459215, 2459216], 2)  # Example Julian days
gspec = read_grid_spec('example_grid_spec.txt')
rp_outvals = evaluate_map(eval_map_file, nc_varname, rawdata, map_grid, nc_grid, fillvalue)

'''
