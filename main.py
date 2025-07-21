import os
import sys
import numpy as np
from scipy.stats import weibull_min, lognorm
from scipy.stats import norm
from netCDF4 import Dataset
import sqlite3
import time
import xarray as xr


# NOTE THE USAGE: python3 main_old.py -annualmax -ncfiles files.list -ncvar swath_peak_wind -rpmaps map.list


xmin = -105.0
xmax = -40.0
ymin = 5.0
ymax = 50.0
xres = 0.1
yres = 0.1


xspan = int(round((xmax - xmin) / xres))
yspan = int(round((ymax - ymin) / yres))


map_grid = {'xmin': -105, 'xmax': -40, 'ymin': 5.0, 'ymax': 50.0,
            'xres': 0.1, 'yres': 0.1,
            'xspan': xspan, 'yspan': yspan}  # Placeholder for xspan and yspan

point_file_list = open('points.list', 'r')


# Define map_grid based on the template from main.f90
def get_nc_grid(nc_file):
    with Dataset(nc_file, 'r') as nc:
        lon = nc.variables['longitude'][:]
        lat = nc.variables['latitude'][:]
        nc_grid = {
            'xmin': lon.min(),
            'xmax': lon.max(),
            'ymin': lat.min(),
            'ymax': lat.max(),
            'xres': lon[1] - lon[0],  # Assuming uniform resolution
            'yres': lat[1] - lat[0],  # Assuming uniform resolution
            'xspan': len(lon),
            'yspan': len(lat)
        }
    return nc_grid





def get_environment_variable(var_name):
    return os.getenv(var_name, '')

def command_argument_count():
    return len(sys.argv) - 1

def getarg(index):
    if index < len(sys.argv):
        return sys.argv[index]
    return ''

def main():
    eval_map = False
    do_points = False
    argin = ['-annualmax', '-ncfiles', 'files.list', '-ncvar', 'swath_peak_wind', '-rpmaps', 'map.list', '-points', 'points.list']
    map_grid_list = 'gridspec.txt'
    BuildData = 'Build Data Placeholder'
    CopyrightData = 'Copyright Data Placeholder'

    print()
    print('TAOS(tm) Hazard Frequency Analysis System')
    print(BuildData)
    print(CopyrightData)
    print()
    # print('Using ', omp_get_max_threads(), ' threads.')
    print()

    mpleshome = get_environment_variable('TAOS_HOME')

    debug = False
    map_grid_xspan = 0
    out_base = 'tstat'
    param_prefix = ''
    ext_numvals = -1
    write_fits = False
    ishurricane = False

    numargs = command_argument_count()
    print(numargs)
    for i in range(1, numargs + 1):
        argin = getarg(i)
        #print('i = ', i)
        #print('argin = ', argin)
        if '-fitext' in argin:
            fit_file = getarg(i + 1)
            analysis_mode = 'MODE_EXTERNAL'
            # write_params = True
        if '-table' in argin:
            sql_tab_name = getarg(i + 1)
            fit_results_file = sql_tab_name
        if '-thres' in argin:
            master_threshold = float(getarg(i + 1))
        if '-ncscale' in argin:
            ncscale = float(getarg(i + 1))
        if '-pre' in argin:
            param_prefix = getarg(i + 1)
        if '-numvals' in argin:
            ext_numvals = int(getarg(i + 1))
        if '-ncfiles' in argin:
            nc_file_list = getarg(i + 1)
        if '-ncvar' in argin:
            nc_varname = getarg(i + 1)
        if '-points' in argin:
            point_file_list = getarg(i + 1)
            analysis_mode = 'MODE_INTERNAL'
            do_points = True
        if '-monthspi' in argin:
            spi_refyy = int(getarg(i + 1))
            spi_refmm = int(getarg(i + 2))
            spi_yy = int(getarg(i + 3))
            spi_mm = int(getarg(i + 4))
            spi_mode = True
            rp_ag_type = 'RP_AGTYPE_SUM'
            rp_ag_mode = 'RP_AGMODE_MONTHLY'
        if '-spi ' in argin:
            spi_mode = True
            spi_query_value = float(getarg(i + 1))
        if '-mfv ' in argin:
            max_fit_vals = float(getarg(i + 1))
        if '-march ' in argin:
            march_size = float(getarg(i + 1))
        if '-eval' in argin:
            eval_map_file = getarg(i + 1)
            eval_map = True
            analysis_mode = 'MODE_INTERNAL'
        if '-rpmaps' in argin:
            rpmap_list = getarg(i + 1)
            analysis_mode = 'MODE_INTERNAL'
            do_maps = True
        if '-annualsum' in argin:
            rp_ag_type = 'RP_AGTYPE_SUM'
            rp_ag_mode = 'RP_AGMODE_ANNUAL'
            # if debug: print('  Computing Annual Max')
        if '-annualmax' in argin:
            rp_ag_type = 'RP_AGTYPE_MAX'
            rp_ag_mode = 'RP_AGMODE_ANNUAL'
            print('  Computing Annual Max')
        if '-monthmax' in argin:
            rp_ag_type = 'RP_AGTYPE_MAX'
            rp_ag_mode = 'RP_AGMODE_MONTHLY'
            target_month = int(getarg(i + 1))
        if '-monthsum' in argin:
            rp_ag_type = 'RP_AGTYPE_SUM'
            rp_ag_mode = 'RP_AGMODE_MONTHLY'
            target_month = int(getarg(i + 1))
        if '-mapgrid' in argin:
            map_grid_list = getarg(i + 1)
            read_grid_spec(map_grid_list, map_grid)
            user_grid = True
        if '-radius' in argin:
            regfit_radius = float(getarg(i + 1))
            regfit = True
        if '-beta' in argin:
            betaonly = True
        if '-debug' in argin:
            debug = True
        if '-optim' in argin:
            optimize_weib = True
        if '-plimits' in argin:
            map_plimits = True
        if '-weibonly' in argin:
            weibull_only = True
        if '-subset' in argin:
            annual_subset = True
        if '-paronly' in argin:
            write_params = True
        if '-writefits' in argin:
            write_fits = True
        if '-solve' in argin:
            fit_file = getarg(i + 1)
            analysis_mode = 'MODE_SOLVE'
            tfield = float(getarg(i + 2))
            weib_fit_param1 = float(getarg(i + 3))
            weib_fit_param2 = float(getarg(i + 4))
            weib_fit_sd_param1 = float(getarg(i + 5))
            weib_fit_sd_param2 = float(getarg(i + 6))
            weib_fit_correl = float(getarg(i + 7))
            weib_fit_abs_zero_frac = float(getarg(i + 8))
            weib_fit_valid = True
        if '-sqlsolve' in argin:
            fit_file = getarg(i + 1)
            analysis_mode = 'MODE_SOLVE_SQL'
            tvarnam = getarg(i + 2)
        if '-stochas' in argin:
            rpmap_list = getarg(i + 1)
            analysis_mode = 'MODE_STOCHAST'
        
    return_periods = [25]
    nc_file_list = []
    for file in os.listdir('/taos/taos_home/tstat_python-main/'):
        if file == 'rp_outvals.nc':
            continue
        if file.endswith('.nc'):
            nc_file_list.append(file)
    #print(nc_file_list)
    nc_varname = 'swath_peak_wind'


    # Do the Dew
    if analysis_mode == 'MODE_EXTERNAL':
        analyze_external_data()
    elif analysis_mode == 'MODE_INTERNAL':
        if do_maps:
            analyze_area(map_grid, nc_file_list, nc_varname, rp_ag_mode, rp_ag_type, return_periods)
        if eval_map:
            evaluate_map()
        if do_points:
            analyze_points(point_file_list, nc_file_list, nc_varname, rp_ag_mode, rp_ag_type)
    elif analysis_mode == 'MODE_SOLVE':
        solve_external_data()
    elif analysis_mode == 'MODE_SOLVE_SQL':
        solve_external_data_sql()
    elif analysis_mode == 'MODE_STOCHAST':
        create_annmax_nc()
        analyze_anmax()
    else:
        print('ERROR: No analysis specified!')




def analyze_external_data(fit_file, fit_results_file, ext_numvals, debug, write_params, spi_mode, spi_query_value, weib_fit, ishurricane):
    def num_records(file_path):
        with open(file_path, 'r') as file:
            return sum(1 for line in file)

    def qsort(arr):
        return np.sort(arr)

    def fit_data(rvals, numvals):
        """
        Fit the data to a Weibull distribution.

        Parameters:
        rvals (numpy array): Array of data values to fit.
        numvals (int): Number of valid data values.

        Returns:
        dict: Fitted parameters of the Weibull distribution.
        """
        # Fit the data to a Weibull distribution
        c, loc, scale = weibull_min.fit(rvals[:numvals], floc=0)

        # Calculate the standard deviation of the parameters
        param1_sd = np.std(rvals[:numvals] ** (1 / c))
        param2_sd = np.std(rvals[:numvals] / scale)

        # Calculate the correlation between the parameters
        correl = np.corrcoef(rvals[:numvals] ** (1 / c), rvals[:numvals] / scale)[0, 1]

        # Calculate the chi-squared value
        chisq = np.sum((rvals[:numvals] - weibull_min(c, loc, scale).cdf(rvals[:numvals])) ** 2)

        # Calculate the fraction of absolute zero values
        abs_zero_frac = np.sum(rvals[:numvals] == 0) / numvals

        # Return the fitted parameters
        weib_fit = {
            'param1': c,
            'param2': scale,
            'sd_param1': param1_sd,
            'sd_param2': param2_sd,
            'correl': correl,
            'chisq': chisq,
            'abs_zero_frac': abs_zero_frac,
            'valid': True
        }

        return weib_fit


    def weibd(param1, param2, value):
        """
        Calculate the cumulative distribution function (CDF) of the Weibull distribution.

        Parameters:
        param1 (float): Shape parameter of the Weibull distribution.
        param2 (float): Scale parameter of the Weibull distribution.
        value (float): The value at which to evaluate the CDF.

        Returns:
        float: The CDF value at the given value.
        """
        return weibull_min.cdf(value, param1, scale=param2)


    def probnorm(rp):
        """
        Calculate the inverse of the cumulative distribution function (CDF) for a normal distribution.

        Parameters:
        rp (float): The probability value for which to calculate the inverse CDF.

        Returns:
        float: The inverse CDF value at the given probability.
        """
        return norm.ppf(rp)


    def generate_weibull_std_plims(rp, mle, p10, p25, p33, p50, p67, p75, p90, p01, p05, p95, p99):
        """
        Generate the standard prediction limits for the Weibull distribution.

        Parameters:
        rp (float): Return period.
        mle (float): Maximum likelihood estimate.
        p10, p25, p33, p50, p67, p75, p90, p01, p05, p95, p99 (float): Prediction limits.

        Returns:
        None
        """
        # Calculate the prediction limits for the Weibull distribution
        p10 = weibull_min.ppf(0.10, rp, scale=mle)
        p25 = weibull_min.ppf(0.25, rp, scale=mle)
        p33 = weibull_min.ppf(0.33, rp, scale=mle)
        p50 = weibull_min.ppf(0.50, rp, scale=mle)
        p67 = weibull_min.ppf(0.67, rp, scale=mle)
        p75 = weibull_min.ppf(0.75, rp, scale=mle)
        p90 = weibull_min.ppf(0.90, rp, scale=mle)
        p01 = weibull_min.ppf(0.01, rp, scale=mle)
        p05 = weibull_min.ppf(0.05, rp, scale=mle)
        p95 = weibull_min.ppf(0.95, rp, scale=mle)
        p99 = weibull_min.ppf(0.99, rp, scale=mle)


    def get_weib_val(rp, weib_fit, flag):
        """
        Calculate the value of the Weibull distribution for a given return period.

        Parameters:
        rp (float): Return period.
        weib_fit (dict): Fitted parameters of the Weibull distribution.
        flag (bool): Flag indicating if the calculation was successful.

        Returns:
        float: The value of the Weibull distribution at the given return period.
        """
        try:
            # Calculate the value of the Weibull distribution
            value = weibull_min.ppf(rp, weib_fit['param1'], scale=weib_fit['param2'])
            flag = True
        except Exception as e:
            print(f"Error calculating Weibull value: {e}")
            value = 0.0
            flag = False

        return value


    def get_all_rp_val(rp, weib_fit, lognorm_fit, ev1l, ev1u, flag):
        """
        Calculate the return period values for different distributions.

        Parameters:
        rp (float): Return period.
        weib_fit (dict): Fitted parameters of the Weibull distribution.
        lognorm_fit (dict): Fitted parameters of the LogNormal distribution.
        ev1l (float): Placeholder for Extreme Value Type I lower bound.
        ev1u (float): Placeholder for Extreme Value Type I upper bound.
        flag (bool): Flag indicating if the calculation was successful.

        Returns:
        tuple: Return period values for Weibull and LogNormal distributions.
        """
        try:
            # Calculate the value of the Weibull distribution
            weib_val = weibull_min.ppf(rp, weib_fit['param1'], scale=weib_fit['param2'])

            # Calculate the value of the LogNormal distribution
            lognorm_val = lognorm.ppf(rp, lognorm_fit['param1'], scale=lognorm_fit['param2'])

            # Placeholder values for Extreme Value Type I distribution
            ev1l = 0.0
            ev1u = 0.0

            flag = True
        except Exception as e:
            print(f"Error calculating return period values: {e}")
            weib_val = 0.0
            lognorm_val = 0.0
            ev1l = 0.0
            ev1u = 0.0
            flag = False

        return weib_val, lognorm_val, ev1l, ev1u


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
                r = float(line.strip())
                rvals[i] = r
                i += 1
            except ValueError:
                continue

    if i < numvals:
        rvals[i:numvals] = 0.0

    if debug:
        print(f"Got {numvals} valid values from external file.")

    svals = qsort(rvals)

    with open('empirical.csv', 'w') as file:
        file.write('p,rpd,val\n')
        for i in range(1, numvals):
            r = (i - 1) / numvals
            rp = 1.0 / (1.0 - r)
            file.write(f"{r:.8f},{rp:.3f},{svals[i]:.5f}\n")

    print("Fitting data . . .")
    fit_data(rvals, numvals)

    if write_params:
        fname = f"{fit_results_file}_params.csv"
        with open(fname, 'w') as file:
            file.write(f"{param_prefix},Weibull,{weib_fit['param1']:.8e},{weib_fit['param2']:.8e},{weib_fit['sd_param1']:.8e},{weib_fit['sd_param2']:.8e},{weib_fit['correl']:.8e},{weib_fit['abs_zero_frac']:.8e},{weib_fit['chisq']:.8e}\n")
            if not weibull_only:
                file.write(f"{param_prefix},LogNormal,{lognorm_fit['param1']:.8e},{lognorm_fit['param2']:.8e},{lognorm_fit['sd_param1']:.8e},{lognorm_fit['sd_param2']:.8e},{lognorm_fit['correl']:.8e},{lognorm_fit['abs_zero_frac']:.8e},{lognorm_fit['chisq']:.8e}\n")
        if not write_fits:
            return

    if spi_mode:
        rp = weibd(weib_fit['param1'], weib_fit['param2'], spi_query_value)
        r = probnorm(rp)
        print(f"SPI: {spi_query_value:.2f},{rp:.7f},{r:.3f}")
    else:
        print("Generating analysis file . . .")
        fname = f"{fit_results_file}_analysis.csv"
        with open(fname, 'w') as file:
            file.write('rp,rpval,mle,p01,p05,p10,p25,p33,p50,p67,p75,p90,p95,p99\n')
            stfac = 1.0
            i = 1
            r = -0.5
            with open('point_multianalysis.csv', 'w') as point_file:
                point_file.write('rp,weib,lnorm\n')
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
                    get_weib_val(rp, r, flag)
                    if not flag:
                        p01 = p05 = p10 = p25 = p33 = p50 = p67 = p75 = p90 = p95 = p99 = mle = r
                    file.write(f"{i/stfac:.1f},{rp:.8f},{mle:.5f},{p01:.5f},{p05:.5f},{p10:.5f},{p25:.5f},{p33:.5f},{p50:.5f},{p67:.5f},{p75:.5f},{p90:.5f},{p95:.5f},{p99:.5f}\n")
                    get_all_rp_val(rp, wb, lno, ev1l, ev1u, flag)
                    point_file.write(f"{i/stfac:.1f},{wb:.5f},{lno:.5f}\n")

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


def resample_to_map_grid(nc_file, nc_varname, map_grid):
    """
    Resample NetCDF data to align with the map_grid.

    Parameters:
    nc_file (str): Path to the NetCDF file.
    nc_varname (str): Variable name in the NetCDF file.
    map_grid (dict): Grid specifications.

    Returns:
    numpy array: Resampled data aligned with map_grid.
    """
    # Open the NetCDF file using xarray
    ds = xr.open_dataset(nc_file, decode_times=False)
    data = ds[nc_varname]

    # Define the target grid based on map_grid
    target_lon = np.linspace(map_grid['xmin'], map_grid['xmax'], map_grid['xspan'])
    target_lat = np.linspace(map_grid['ymin'], map_grid['ymax'], map_grid['yspan'])

    # Interpolate the data to the target grid
    data_resampled = data.interp(
        longitude=target_lon,
        latitude=target_lat,
        method="nearest"
    )

    ds.close()

    # Return the resampled data as a numpy array
    return data_resampled.values

def analyze_area(map_grid, nc_files, nc_varname, rp_ag_mode, rp_ag_type, return_periods, debug=False, regfit=False, regfit_radius=0, spi_mode=False, spi_refyy=0, spi_refmm=0, base_year=1950, end_year=2023):
    """
    Analyze area for return period maps.

    Parameters:
    map_grid (dict): Grid specifications.
    nc_files (list): List of NetCDF files.
    nc_varname (str): Variable name in NetCDF files.
    rp_ag_mode (str): Aggregation mode.
    rp_ag_type (str): Aggregation type.
    return_periods (list): List of return periods.
    debug (bool): Debug flag.
    regfit (bool): Regional fit flag.
    regfit_radius (int): Regional fit radius.
    spi_mode (bool): SPI mode flag.
    spi_refyy (int): SPI reference year.
    spi_refmm (int): SPI reference month.
    base_year (int): Base year.
    end_year (int): End year.

    Returns:
    None
    """
    def fit_data(cell_vals, num_vals):
        c, loc, scale = weibull_min.fit(cell_vals[:num_vals], floc=0)
        return {'param1': c, 'param2': scale, 'valid': True}

    def generate_weibull_mle(rpval, weib_fit):
        return weibull_min.ppf(rpval, weib_fit['param1'], scale=weib_fit['param2'])

    num_rvals = len(nc_files)
    if rp_ag_mode == 'RP_AGMODE_MONTHLY':
        num_rvals = (end_year - base_year) * 12
    elif rp_ag_mode == 'RP_AGMODE_ANNUAL':
        num_rvals = end_year - base_year

    # Initialize rawdata array to store resampled data
    rawdata = np.zeros((map_grid['xspan'], map_grid['yspan'], num_rvals))

    for i, nc_file in enumerate(nc_files):
        if nc_file == 'rp_outvals_xarray.nc':
            continue
        # get the ncfile year
        nc_year = nc_file[11:15]
        i = int(nc_year) - base_year # Adjust the index based on base year (1950)
        print(nc_file)
        # Resample the NetCDF data to align with map_grid
        resampled_data = resample_to_map_grid(nc_file, nc_varname, map_grid)

        # Aggregate the resampled data into rawdata
        for x in range(map_grid['xspan']):
            for y in range(map_grid['yspan']):
                rval = resampled_data[y, x]  # Note: y corresponds to latitude, x to longitude
                if rp_ag_mode == 'RP_AGMODE_NONE':
                    idx = i
                elif rp_ag_mode == 'RP_AGMODE_ANNUAL':
                    idx = i // 12
                elif rp_ag_mode == 'RP_AGMODE_MONTHLY':
                    idx = i 
                else:
                    idx = 0  # Default index if mode is unexpected

                # Ensure idx is within bounds
                #if idx >= num_rvals:
                #    print(f"Skipping out-of-bounds index: idx={idx}, num_rvals={num_rvals}")
                #    continue

                if rp_ag_type == 'RP_AGTYPE_SUM':
                    rawdata[x, y, idx] += rval
                elif rp_ag_type == 'RP_AGTYPE_MAX':
                    rawdata[x, y, idx] = max(rawdata[x, y, idx], rval)

    if spi_mode:
        rawdata *= 3937.0

    # Fit data to Weibull distribution
    map_fit = np.zeros((map_grid['xspan'], map_grid['yspan']), dtype=object)
    for x in range(map_grid['xspan']):
        for y in range(map_grid['yspan']):
            cell_vals = rawdata[x, y, :]
            if np.max(cell_vals) != 0:
                map_fit[x, y] = fit_data(cell_vals, num_rvals)

    rp_outvals = np.zeros((map_grid['xspan'], map_grid['yspan']))
    for rp in return_periods:
        rpval = 1.0 - (1.0 / rp)
        for x in range(map_grid['xspan']):
            for y in range(map_grid['yspan']):
                if map_fit[x, y] and map_fit[x, y]['valid']:
                    weib_fit = map_fit[x, y]
                    mle = generate_weibull_mle(rpval, weib_fit)
                    rp_outvals[x, y] = mle

    print(np.max(rp_outvals))


    # Save rp_outvals to a NetCDF file
    output_ncfile = "rp_outvals.nc"
    with Dataset(output_ncfile, "w", format="NETCDF4") as ncfile:
        # Create dimensions
        ncfile.createDimension("x", map_grid['xspan'])
        ncfile.createDimension("y", map_grid['yspan'])

        # Create variables
        x_var = ncfile.createVariable("x", "f4", ("x",))
        y_var = ncfile.createVariable("y", "f4", ("y",))
        rp_outvals_var = ncfile.createVariable("rp_outvals", "f4", ("x", "y"))

        # Assign data to variables
        x_var[:] = np.linspace(map_grid['xmin'], map_grid['xmax'], map_grid['xspan'])
        y_var[:] = np.linspace(map_grid['ymin'], map_grid['ymax'], map_grid['yspan'])
        rp_outvals_var[:, :] = rp_outvals

        # Add metadata
        x_var.units = "degrees_east"
        y_var.units = "degrees_north"
        rp_outvals_var.units = "return_period_values"
        rp_outvals_var.description = "Return period values for the grid"

        print(f"Saved rp_outvals to {output_ncfile}")



    # Correctly calculate x_var and y_var
#    x_var = np.linspace(map_grid['xmin'], map_grid['xmax'], rp_outvals.shape[0])
#    y_var = np.linspace(map_grid['ymin'], map_grid['ymax'], rp_outvals.shape[1])

    # Create xarray DataArray and save to NetCDF
#    data_xr = xr.DataArray(rp_outvals.T, coords={'y': y_var, 'x': x_var}, dims=["y", "x"])
#    dataset = xr.Dataset({"rp_vals": data_xr})
#    dataset.to_netcdf('rp_outvals_xarray.nc')






def evaluate_map(eval_map_file, nc_varname, rawdata, map_grid, fillvalue, debug=False):
    """
    Evaluate a map by comparing the values in a NetCDF file against the raw data and calculating the percentile rank for each grid cell.

    Parameters:
    eval_map_file (str): Path to the NetCDF file to evaluate.
    nc_varname (str): Variable name in the NetCDF file.
    rawdata (numpy array): Raw data array.
    map_grid (dict): Grid specifications.
    fillvalue (float): Fill value for missing data.
    debug (bool): Debug flag.

    Returns:
    numpy array: Percentile rank values for each grid cell.
    """
    def get_nc_vals(nc_file, varname):
        with Dataset(nc_file, 'r') as nc:
            return nc.variables[varname][:]

    num_vals = rawdata.shape[2]
    cell_vals = np.zeros(num_vals)
    rp_outvals = np.zeros((map_grid['xspan'], map_grid['yspan']))

    nc_vals = get_nc_vals(eval_map_file, nc_varname)

    for xp in range(map_grid['xspan']):
        for yp in range(map_grid['yspan']):
            cell_vals = rawdata[xp, yp, :]
            cell_vals.sort()

            if cell_vals[-1] >= 1e20:
                pval = -1
            else:
                rval = nc_vals[xp, yp]
                if rval == fillvalue:
                    pval = -1
                else:
                    pval = np.sum(rval > cell_vals) / num_vals

            rp_outvals[xp, yp] = pval

    return rp_outvals

''''''

def analyze_points(point_file_list, nc_files, nc_varname, rp_ag_mode, rp_ag_type, base_year=1950, end_year=2023, ext_numvals=1000, debug=False, write_params=False, weibull_only=True, annual_subset=False):
    """
    Analyze points for return period maps.

    Parameters:
    point_file_list (str): Path to the point file list.
    nc_files (list): List of NetCDF files.
    nc_varname (str): Variable name in NetCDF files.
    rp_ag_mode (str): Aggregation mode.
    rp_ag_type (str): Aggregation type.
    base_year (int): Base year.
    end_year (int): End year.
    ext_numvals (int): Number of external values.
    debug (bool): Debug flag.
    write_params (bool): Flag to write parameters.
    weibull_only (bool): Flag to use only Weibull distribution.
    annual_subset (bool): Flag to use annual subset.

    Returns:
    None
    """
    def read_point_list(point_file_list):
        points = []
        with open(point_file_list, 'r') as file:
            for line in file:
                if line.startswith('#'):
                    continue
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    name = parts[0]
                    lat = float(parts[1])
                    lon = float(parts[2])
                    points.append((name, lat, lon))
        return points

    def get_nc_vals(nc_file, varname):
        with Dataset(nc_file, 'r') as nc:
            return nc.variables[varname][:]

    def ll_to_gridxy(nc_grid, lat, lon):
        x = int((lon - nc_grid['xmin']) / nc_grid['xres'])
        y = int((lat - nc_grid['ymin']) / nc_grid['yres'])
        ongrid = (0 <= x < nc_grid['xspan']) and (0 <= y < nc_grid['yspan'])
        return x, y, ongrid

    def fit_data(rvals, numvals):
        c, loc, scale = weibull_min.fit(rvals[:numvals], floc=0)
        return {'param1': c, 'param2': scale, 'valid': True}

    points = read_point_list(point_file_list)
    num_points = len(points)
    num_ncfiles = len(nc_files)
    num_rvals = num_ncfiles
    if rp_ag_mode == 'RP_AGMODE_MONTHLY':
        num_rvals = (end_year - base_year) * 12
    elif rp_ag_mode == 'RP_AGMODE_ANNUAL':
        num_rvals = end_year - base_year

    pt_vals = np.zeros((num_points, num_rvals))


    for i, nc_file in enumerate(nc_files):
        if nc_file[0:6] != "taostc":
            continue
        nc_year = nc_file[11:15]
        i = int(nc_year) - base_year
        nc_vals = get_nc_vals(nc_file, nc_varname)
        nc_grid = get_nc_grid(nc_file)
        for j, (name, lat, lon) in enumerate(points):
            x, y, ongrid = ll_to_gridxy(nc_grid, lat, lon)
            rval = 0.0
            if ongrid:
                rval = nc_vals[y, x]
            if rp_ag_mode == 'RP_AGMODE_NONE':
                idx = i
            elif rp_ag_mode == 'RP_AGMODE_ANNUAL':
                idx = i // 12
            elif rp_ag_mode == 'RP_AGMODE_MONTHLY':
                idx = i
            if rp_ag_type == 'RP_AGTYPE_SUM':
                pt_vals[j, idx] += rval
            elif rp_ag_type == 'RP_AGTYPE_MAX':
                pt_vals[j, idx] = max(pt_vals[j, idx], rval)

    for j, (name, lat, lon) in enumerate(points):
        rvals = pt_vals[j, :]
        svals = np.sort(rvals)
        numanvals = len(rvals)

        if write_params:
            with open('point_fit_params.csv', 'a') as file:
                weib_fit = fit_data(rvals, numanvals)
                file.write(f"{name},Weibull,{weib_fit['param1']},{weib_fit['param2']}\n")
                if not weibull_only:
                    lognorm_fit = lognorm.fit(rvals[:numanvals])
                    file.write(f"{name},LogNormal,{lognorm_fit[0]},{lognorm_fit[2]}\n")
        else:
            with open(f"{name}_empirical.csv", 'w') as file:
                file.write('p,rpd,val\n')
                for i in range(1, numanvals):
                    r = (i - 1) / numanvals
                    rp = 1.0 / (1.0 - r)
                    file.write(f"{r},{rp},{svals[i]}\n")

            with open(f"{name}_analysis.csv", 'w') as file:
                file.write('rp,rpval,mle,p10,p25,p33,p50,p67,p75,p90\n')
                for i in range(20, numanvals * 50):
                    rp = 1.0 - (1.0 / (i / 10.0))
                    weib_fit = fit_data(rvals, numanvals)
                    print(weib_fit)
                    print(rvals)
                    mle = weibull_min.ppf(rp, weib_fit['param1'], scale=weib_fit['param2'])
                    
                    # Dynamically calculate quantiles based on the current rp value
                    p10 = weibull_min.ppf(0.10, weib_fit['param1'], scale=weib_fit['param2'])
                    p25 = weibull_min.ppf(0.25, weib_fit['param1'], scale=weib_fit['param2'])
                    p33 = weibull_min.ppf(0.33, weib_fit['param1'], scale=weib_fit['param2'])
                    p50 = weibull_min.ppf(0.50, weib_fit['param1'], scale=weib_fit['param2'])
                    p67 = weibull_min.ppf(0.67, weib_fit['param1'], scale=weib_fit['param2'])
                    p75 = weibull_min.ppf(0.75, weib_fit['param1'], scale=weib_fit['param2'])
                    p90 = weibull_min.ppf(0.90, weib_fit['param1'], scale=weib_fit['param2'])
                    
                    print(weib_fit['param1'])
                    print(weib_fit['param2'])
                    

                    # Write the dynamically calculated values to the file
                    file.write(f"{i / 10.0},{rp},{mle},{p10},{p25},{p33},{p50},{p67},{p75},{p90}\n")

                
''''''

import numpy as np
from scipy.stats import weibull_min, lognorm
from netCDF4 import Dataset



def analyze_points(point_file_list, nc_files, nc_varname, rp_ag_mode, rp_ag_type, base_year=1950, end_year=2022, ext_numvals=1, debug=False, write_params=False, weibull_only=True, annual_subset=False, regfit=False, regfit_radius=0):
    """
    Analyze points for return period maps.

    Parameters:
    point_file_list (str): Path to the point file list.
    nc_files (list): List of NetCDF files.
    nc_varname (str): Variable name in NetCDF files.
    rp_ag_mode (str): Aggregation mode.
    rp_ag_type (str): Aggregation type.
    base_year (int): Base year.
    end_year (int): End year.
    ext_numvals (int): Number of external values.
    debug (bool): Debug flag.
    write_params (bool): Flag to write parameters.
    weibull_only (bool): Flag to use only Weibull distribution.
    annual_subset (bool): Flag to use annual subset.
    regfit (bool): Regional fit flag.
    regfit_radius (int): Regional fit radius.

    Returns:
    None
    """
    def read_point_list(point_file_list):
        points = []
        with open(point_file_list, 'r') as file:
            for line in file:
                if line.startswith('#'):
                    continue
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    name = parts[0]
                    lat = float(parts[1])
                    lon = float(parts[2])
                    points.append((name, lat, lon))
        return points

    def get_nc_vals(nc_file, varname):
        with Dataset(nc_file, 'r') as nc:
            return nc.variables[varname][:]

    def ll_to_gridxy(nc_grid, lat, lon):
        x = int((lon - nc_grid['xmin']) / nc_grid['xres'])
        y = int((lat - nc_grid['ymin']) / nc_grid['yres'])
        ongrid = (0 <= x < nc_grid['xspan']) and (0 <= y < nc_grid['yspan'])
        return x, y, ongrid

    def fit_data(rvals, numvals):
        c, loc, scale = weibull_min.fit(rvals[:numvals], floc=0)
        return {'param1': c, 'param2': scale, 'valid': True}

    def qsort(arr):
        return np.sort(arr)

    # Read points
    points = read_point_list(point_file_list)
    num_points = len(points)
    num_ncfiles = len(nc_files)
    num_rvals = num_ncfiles
    if rp_ag_mode == 'RP_AGMODE_MONTHLY':
        num_rvals = (end_year - base_year) * 12
    elif rp_ag_mode == 'RP_AGMODE_ANNUAL':
        num_rvals = end_year - base_year

    pt_vals = np.zeros((num_points, num_rvals))

    # Process NetCDF files
    for i, nc_file in enumerate(nc_files):
        if nc_file[0:6] != "taostc":
            continue
        nc_year = nc_file[11:15]
        i = int(nc_year) - base_year
        if debug:
            print(f"Processing {nc_file}")
        nc_vals = get_nc_vals(nc_file, nc_varname)
        nc_grid = get_nc_grid(nc_file)
        for j, (name, lat, lon) in enumerate(points):
            x, y, ongrid = ll_to_gridxy(nc_grid, lat, lon)
            rval = 0.0
            if ongrid:
                if regfit:
                    rval = 0.0
                    for xz in range(max(0, x - regfit_radius), min(nc_grid['xspan'], x + regfit_radius + 1)):
                        for yz in range(max(0, y - regfit_radius), min(nc_grid['yspan'], y + regfit_radius + 1)):
                            rval = max(rval, nc_vals[yz, xz])
                else:
                    rval = nc_vals[y, x]

            if rp_ag_mode == 'RP_AGMODE_NONE':
                idx = i
            elif rp_ag_mode == 'RP_AGMODE_ANNUAL':
                idx = i // 12
            elif rp_ag_mode == 'RP_AGMODE_MONTHLY':
                idx = i

            if rp_ag_type == 'RP_AGTYPE_SUM':
                pt_vals[j, idx] += rval
            elif rp_ag_type == 'RP_AGTYPE_MAX':
                pt_vals[j, idx] = max(pt_vals[j, idx], rval)

    # Fit data and write results
    for j, (name, lat, lon) in enumerate(points):
        rvals = pt_vals[j, :]
        svals = qsort(rvals)
        numanvals = len(rvals)

        if write_params:
            with open('point_fit_params.csv', 'a') as file:
                weib_fit = fit_data(rvals, numanvals)
                file.write(f"{name},Weibull,{weib_fit['param1']},{weib_fit['param2']}\n")
                if not weibull_only:
                    lognorm_fit = lognorm.fit(rvals[:numanvals])
                    file.write(f"{name},LogNormal,{lognorm_fit[0]},{lognorm_fit[2]}\n")

        else:
            with open(f"{name}_empirical.csv", 'w') as file:
                file.write('p,rpd,val\n')
                for i in range(1, numanvals):
                    r = (i - 1) / numanvals
                    rp = 1.0 / (1.0 - r)
                    file.write(f"{r},{rp},{svals[i]}\n")

            with open(f"{name}_analysis.csv", 'w') as file:
                file.write('rp,rpval,mle,p10,p25,p33,p50,p67,p75,p90\n')
                for i in range(20, numanvals * 50):
                    rp = 1.0 - (1.0 / (i / 10.0))
                    weib_fit = fit_data(rvals, numanvals)
                    mle = weibull_min.ppf(rp, weib_fit['param1'], scale=weib_fit['param2'])
                    p10 = weibull_min.ppf(0.10, weib_fit['param1'], scale=weib_fit['param2'])
                    p25 = weibull_min.ppf(0.25, weib_fit['param1'], scale=weib_fit['param2'])
                    p33 = weibull_min.ppf(0.33, weib_fit['param1'], scale=weib_fit['param2'])
                    p50 = weibull_min.ppf(0.50, weib_fit['param1'], scale=weib_fit['param2'])
                    p67 = weibull_min.ppf(0.67, weib_fit['param1'], scale=weib_fit['param2'])
                    p75 = weibull_min.ppf(0.75, weib_fit['param1'], scale=weib_fit['param2'])
                    p90 = weibull_min.ppf(0.90, weib_fit['param1'], scale=weib_fit['param2'])
                    file.write(f"{i / 10.0},{rp},{mle},{p10},{p25},{p33},{p50},{p67},{p75},{p90}\n")

    print("Point processing complete.")





def solve_external_data(fit_file, tfield, weib_fit, debug=False):
    """
    Solve external data by fitting it to a Weibull distribution and calculating return period values.

    Parameters:
    fit_file (str): Path to the external data file.
    tfield (float): Threshold field value.
    weib_fit (dict): Fitted parameters of the Weibull distribution.
    debug (bool): Debug flag.

    Returns:
    None
    """
    def num_records(file_path):
        with open(file_path, 'r') as file:
            return sum(1 for line in file)

    def fit_data(rvals, numvals):
        c, loc, scale = weibull_min.fit(rvals[:numvals], floc=0)
        return {'param1': c, 'param2': scale, 'valid': True}

    def weibd(param1, param2, value):
        return weibull_min.cdf(value, param1, scale=param2)

    def generate_weibull_std_plims(rp, weib_fit):
        p10 = weibull_min.ppf(0.10, weib_fit['param1'], scale=weib_fit['param2'])
        p25 = weibull_min.ppf(0.25, weib_fit['param1'], scale=weib_fit['param2'])
        p33 = weibull_min.ppf(0.33, weib_fit['param1'], scale=weib_fit['param2'])
        p50 = weibull_min.ppf(0.50, weib_fit['param1'], scale=weib_fit['param2'])
        p67 = weibull_min.ppf(0.67, weib_fit['param1'], scale=weib_fit['param2'])
        p75 = weibull_min.ppf(0.75, weib_fit['param1'], scale=weib_fit['param2'])
        p90 = weibull_min.ppf(0.90, weib_fit['param1'], scale=weib_fit['param2'])
        p01 = weibull_min.ppf(0.01, weib_fit['param1'], scale=weib_fit['param2'])
        p05 = weibull_min.ppf(0.05, weib_fit['param1'], scale=weib_fit['param2'])
        p95 = weibull_min.ppf(0.95, weib_fit['param1'], scale=weib_fit['param2'])
        p99 = weibull_min.ppf(0.99, weib_fit['param1'], scale=weib_fit['param2'])
        return p10, p25, p33, p50, p67, p75, p90, p01, p05, p95, p99

    print(f"Solving external data file {fit_file}")
    if not os.path.exists(fit_file):
        print(f"EPIC FAIL: {fit_file} does not exist.")
        return

    numvals = num_records(fit_file)
    rvals = np.zeros(numvals)

    with open(fit_file, 'r') as file:
        i = 0
        for line in file:
            try:
                r = float(line.strip())
                rvals[i] = r
                i += 1
            except ValueError:
                continue

    if i < numvals:
        rvals[i:numvals] = 0.0

    if debug:
        print(f"Got {numvals} valid values from external file.")

    weib_fit = fit_data(rvals, numvals)

    rp = weibd(weib_fit['param1'], weib_fit['param2'], tfield)
    p10, p25, p33, p50, p67, p75, p90, p01, p05, p95, p99 = generate_weibull_std_plims(rp, weib_fit)

    print(f"Return period values for threshold {tfield}:")
    print(f"p10: {p10}, p25: {p25}, p33: {p33}, p50: {p50}, p67: {p67}, p75: {p75}, p90: {p90}, p01: {p01}, p05: {p05}, p95: {p95}, p99: {p99}")


def solve_external_data_sql(fit_file, tvarnam, weib_fit, debug=False):
    """
    Solve external data by fitting it to a Weibull distribution and calculating return period values, with SQL operations.

    Parameters:
    fit_file (str): Path to the external data file.
    tvarnam (str): Target variable name for SQL operations.
    weib_fit (dict): Fitted parameters of the Weibull distribution.
    debug (bool): Debug flag.

    Returns:
    None
    """
    def num_records(file_path):
        with open(file_path, 'r') as file:
            return sum(1 for line in file)

    def fit_data(rvals, numvals):
        c, loc, scale = weibull_min.fit(rvals[:numvals], floc=0)
        return {'param1': c, 'param2': scale, 'valid': True}

    def weibd(param1, param2, value):
        return weibull_min.cdf(value, param1, scale=param2)

    def generate_weibull_std_plims(rp, weib_fit):
        p10 = weibull_min.ppf(0.10, weib_fit['param1'], scale=weib_fit['param2'])
        p25 = weibull_min.ppf(0.25, weib_fit['param1'], scale=weib_fit['param2'])
        p33 = weibull_min.ppf(0.33, weib_fit['param1'], scale=weib_fit['param2'])
        p50 = weibull_min.ppf(0.50, weib_fit['param1'], scale=weib_fit['param2'])
        p67 = weibull_min.ppf(0.67, weib_fit['param1'], scale=weib_fit['param2'])
        p75 = weibull_min.ppf(0.75, weib_fit['param1'], scale=weib_fit['param2'])
        p90 = weibull_min.ppf(0.90, weib_fit['param1'], scale=weib_fit['param2'])
        p01 = weibull_min.ppf(0.01, weib_fit['param1'], scale=weib_fit['param2'])
        p05 = weibull_min.ppf(0.05, weib_fit['param1'], scale=weib_fit['param2'])
        p95 = weibull_min.ppf(0.95, weib_fit['param1'], scale=weib_fit['param2'])
        p99 = weibull_min.ppf(0.99, weib_fit['param1'], scale=weib_fit['param2'])
        return p10, p25, p33, p50, p67, p75, p90, p01, p05, p95, p99

    print(f"Solving external data file {fit_file} with SQL operations")
    if not os.path.exists(fit_file):
        print(f"EPIC FAIL: {fit_file} does not exist.")
        return

    numvals = num_records(fit_file)
    rvals = np.zeros(numvals)

    with open(fit_file, 'r') as file:
        i = 0
        for line in file:
            try:
                r = float(line.strip())
                rvals[i] = r
                i += 1
            except ValueError:
                continue

    if i < numvals:
        rvals[i:numvals] = 0.0

    if debug:
        print(f"Got {numvals} valid values from external file.")

    weib_fit = fit_data(rvals, numvals)

    # Connect to the SQLite database
    conn = sqlite3.connect('results.db')
    cursor = conn.cursor()

    # Create a table if it doesn't exist
    cursor.execute(f"""
    CREATE TABLE IF NOT EXISTS {tvarnam} (
        rp REAL,
        rpval REAL,
        mle REAL,
        p01 REAL,
        p05 REAL,
        p10 REAL,
        p25 REAL,
        p33 REAL,
        p50 REAL,
        p67 REAL,
        p75 REAL,
        p90 REAL,
        p95 REAL,
        p99 REAL
    )
    """)

    rp = weibd(weib_fit['param1'], weib_fit['param2'], tfield)
    p10, p25, p33, p50, p67, p75, p90, p01, p05, p95, p99 = generate_weibull_std_plims(rp, weib_fit)

    # Insert the results into the table
    cursor.execute(f"""
    INSERT INTO {tvarnam} (rp, rpval, mle, p01, p05, p10, p25, p33, p50, p67, p75, p90, p95, p99)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (rp, rp, p50, p01, p05, p10, p25, p33, p50, p67, p75, p90, p95, p99))

    # Commit the transaction and close the connection
    conn.commit()
    conn.close()

    print(f"Return period values for threshold {tfield} have been saved to the database.")


def max_nc(nc_files, nc_varname, map_grid, out_base, base_year, end_year, debug=False):
    """
    Create an annual maximum dataset from NetCDF files.

    Parameters:
    nc_files (list): List of NetCDF files.
    nc_varname (str): Variable name in NetCDF files.
    map_grid (dict): Grid specifications.
    out_base (str): Output base name.
    base_year (int): Base year.
    end_year (int): End year.
    debug (bool): Debug flag.

    Returns:
    None
    """
    def get_nc_vals(nc_file, varname):
        with Dataset(nc_file, 'r') as nc:
            return nc.variables[varname][:]

    def gridxy_to_ll(map_grid, x, y):
        lat = map_grid['ymin'] + y * map_grid['yres']
        lon = map_grid['xmin'] + x * map_grid['xres']
        return lat, lon

    def ll_to_gridxy(nc_grid, lat, lon):
        x = int((lon - nc_grid['xmin']) / nc_grid['xres'])
        y = int((lat - nc_grid['ymin']) / nc_grid['yres'])
        ongrid = (0 <= x < nc_grid['xspan']) and (0 <= y < nc_grid['yspan'])
        return x, y, ongrid

    nyears = end_year - base_year
    anmax = np.zeros((map_grid['xspan'], map_grid['yspan']))

    ncfname = f"{out_base}_annmax.nc"
    with Dataset(ncfname, 'w', format='NETCDF4') as ncid:
        ncid.createDimension('latitude', map_grid['yspan'])
        ncid.createDimension('longitude', map_grid['xspan'])
        ncid.createDimension('year', nyears)

        latitudes = ncid.createVariable('latitude', 'f4', ('latitude',))
        longitudes = ncid.createVariable('longitude', 'f4', ('longitude',))
        annmax_var = ncid.createVariable('annmax', 'f4', ('year', 'latitude', 'longitude',))

        latitudes.units = 'degrees_north'
        longitudes.units = 'degrees_east'

        latitudes[:] = np.linspace(map_grid['ymin'], map_grid['ymax'], map_grid['yspan'])
        longitudes[:] = np.linspace(map_grid['xmin'], map_grid['xmax'], map_grid['xspan'])

        for yr in range(base_year + 1, end_year + 1):
            stime = time.time()
            nevents = 0
            if debug:
                print(f"Processing {yr}")
            anmax.fill(0)

            for nc_file in nc_files:
                nc_vals = get_nc_vals(nc_file, nc_varname)
                for x in range(map_grid['xspan']):
                    for y in range(map_grid['yspan']):
                        lat, lon = gridxy_to_ll(map_grid, x, y)
                        xp, yp, ongrid = ll_to_gridxy(map_grid, lat, lon)
                        rval = 0.0
                        if ongrid:
                            rval = nc_vals[xp, yp]
                        if rval > anmax[x, y]:
                            anmax[x, y] = rval

            annmax_var[yr - base_year - 1, :, :] = anmax
            etime = time.time() - stime
            if debug:
                print(f"{yr} had {nevents} max {np.max(anmax)} time {etime}")


def analyze_anmax(nc_file, map_grid, return_periods, out_base, debug=False):
    """
    Analyze annual maximum dataset to generate return period maps.

    Parameters:
    nc_file (str): Path to the NetCDF file containing annual maximum data.
    map_grid (dict): Grid specifications.
    return_periods (list): List of return periods.
    out_base (str): Output base name.
    debug (bool): Debug flag.

    Returns:
    None
    """
    def get_nc_vals(nc_file, varname):
        with Dataset(nc_file, 'r') as nc:
            return nc.variables[varname][:]

    def fit_data(rvals, numvals):
        c, loc, scale = weibull_min.fit(rvals[:numvals], floc=0)
        return {'param1': c, 'param2': scale, 'valid': True}

    def generate_weibull_mle(rpval, weib_fit):
        return weibull_min.ppf(rpval, weib_fit['param1'], scale=weib_fit['param2'])

    def generate_weibull_std_plims(rpval, weib_fit):
        p10 = weibull_min.ppf(0.10, weib_fit['param1'], scale=weib_fit['param2'])
        p25 = weibull_min.ppf(0.25, weib_fit['param1'], scale=weib_fit['param2'])
        p33 = weibull_min.ppf(0.33, weib_fit['param1'], scale=weib_fit['param2'])
        p50 = weibull_min.ppf(0.50, weib_fit['param1'], scale=weib_fit['param2'])
        p67 = weibull_min.ppf(0.67, weib_fit['param1'], scale=weib_fit['param2'])
        p75 = weibull_min.ppf(0.75, weib_fit['param1'], scale=weib_fit['param2'])
        p90 = weibull_min.ppf(0.90, weib_fit['param1'], scale=weib_fit['param2'])
        p01 = weibull_min.ppf(0.01, weib_fit['param1'], scale=weib_fit['param2'])
        p05 = weibull_min.ppf(0.05, weib_fit['param1'], scale=weib_fit['param2'])
        p95 = weibull_min.ppf(0.95, weib_fit['param1'], scale=weib_fit['param2'])
        p99 = weibull_min.ppf(0.99, weib_fit['param1'], scale=weib_fit['param2'])
        return p10, p25, p33, p50, p67, p75, p90, p01, p05, p95, p99

    print(f"Analyzing annual maximum dataset from {nc_file}")
    with Dataset(nc_file, 'r') as ncid:
        annmax = ncid.variables['annmax'][:]
        nyears = annmax.shape[0]

        rp_outvals = np.zeros((map_grid['xspan'], map_grid['yspan'], len(return_periods)))

        for x in range(map_grid['xspan']):
            for y in range(map_grid['yspan']):
                maxvals = annmax[:, y, x]
                maxvals = maxvals[maxvals < 9e30]  # Filter out invalid values
                if len(maxvals) == 0:
                    continue

                weib_fit = fit_data(maxvals, len(maxvals))

                for i, rp in enumerate(return_periods):
                    rpval = 1.0 - (1.0 / rp)
                    mle = generate_weibull_mle(rpval, weib_fit)
                    rp_outvals[x, y, i] = mle

        # Save rp_outvals to a NetCDF file or other output format as needed
        rpncid = Dataset(f"{out_base}_rpmaps.nc", 'w', format='NETCDF4')
        rpncid.createDimension('latitude', map_grid['yspan'])
        rpncid.createDimension('longitude', map_grid['xspan'])
        rpncid.createDimension('return_period', len(return_periods))

        latitudes = rpncid.createVariable('latitude', 'f4', ('latitude',))
        longitudes = rpncid.createVariable('longitude', 'f4', ('longitude',))
        return_period_var = rpncid.createVariable('return_period', 'f4', ('return_period',))
        rpmap_var = rpncid.createVariable('rpmap', 'f4', ('return_period', 'latitude', 'longitude',))

        latitudes.units = 'degrees_north'
        longitudes.units = 'degrees_east'
        return_period_var.units = 'years'

        latitudes[:] = np.linspace(map_grid['ymin'], map_grid['ymax'], map_grid['yspan'])
        longitudes[:] = np.linspace(map_grid['xmin'], map_grid['xmax'], map_grid['xspan'])
        return_period_var[:] = return_periods
        rpmap_var[:, :, :] = rp_outvals.transpose(2, 1, 0)

        rpncid.close()

        print(f"Return period maps saved to {out_base}_rpmaps.nc")


def read_grid_spec(fname, gspec):
    """
    Read grid specifications from a file and populate the gspec structure.

    Parameters:
    fname (str): Path to the grid specification file.
    gspec (dict): Grid specification structure to populate.

    Returns:
    None
    """
    import csv

    print("Reading grid specification file . . .")
    try:
        with open(fname, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if not row or row[0].startswith('#'):
                    continue
                key = row[0].strip().lower()
                if key == "bounds":
                    gspec['ymin'] = float(row[1])
                    gspec['xmin'] = float(row[2])
                    gspec['ymax'] = float(row[3])
                    gspec['xmax'] = float(row[4])
                elif key == "res":
                    gspec['xres'] = float(row[1])
                    gspec['yres'] = float(row[2])
                    gspec['xspan'] = int((gspec['xmax'] - gspec['xmin']) / gspec['xres'])
                    gspec['yspan'] = int((gspec['ymax'] - gspec['ymin']) / gspec['yres'])
                elif key == "span":
                    gspec['xspan'] = int(row[1])
                    gspec['yspan'] = int(row[2])
                    gspec['xres'] = (gspec['xmax'] - gspec['xmin']) / gspec['xspan']
                    gspec['yres'] = (gspec['ymax'] - gspec['ymin']) / gspec['yspan']
    except FileNotFoundError:
        print(f"EPIC FAIL: {fname} does not exist.")
        return

    print(gspec)

if __name__ == '__main__':
    main()

main()
