#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 17:07:01 2025

@author: agastraa
"""

import numpy as np

class FitType:
    def __init__(self):
        self.param1 = 0
        self.param2 = 0
        self.sd_param1 = 0
        self.sd_param2 = 0
        self.correl = 0
        self.chisq = 0
        self.valid = False
        self.zf_nvals = 0
        self.zf_mean = 0
        self.zf_stdev = 0
        self.abs_zero_frac = 0
        self.thres_used = 0
        self.minval = 0

class NewFitType:
    def __init__(self):
        self.param1 = 0
        self.param2 = 0
        self.sd_param1 = 0
        self.sd_param2 = 0
        self.correl = 0
        self.chisq = 0
        self.nvals = 0
        self.nfit = 0
        self.nzero = 0
        self.thres_rval = 0
        self.thres_pval = 0
        self.minval = 0
        self.have_ovals = False
        self.ovals = []

class RPMapModule:
    DISTRIB_EMPIRIC = 0
    DISTRIB_WEIBULL = 1
    DISTRIB_LOGNORM = 2
    DISTRIB_EVT1MAX = 3
    DISTRIB_EVT1MIN = 4
    DISTRIB_EVT1COM = 5

    MODE_NONE = 0
    MODE_EXTERNAL = 1
    MODE_INTERNAL = 2
    MODE_SOLVE = 3
    MODE_SOLVE_SQL = 4
    MODE_STOCHAST = 5

    RP_AGTYPE_NONE = 0
    RP_AGTYPE_SUM = 1
    RP_AGTYPE_MAX = 2

    RP_AGMODE_NONE = 0
    RP_AGMODE_ANNUAL = 1
    RP_AGMODE_MONTHLY = 2

    def __init__(self):
        # General settings
        self.debug = False
        self.UTMP = 200
        self.USQL = 201
        self.UWYR = 202

        self.analysis_mode = self.MODE_NONE
        self.do_points = False
        self.do_maps = False
        self.write_fits = False
        self.regfit = False
        self.regfit_radius = 10

        self.ishurricane = True
        self.write_params = False
        self.param_prefix = ""

        self.betaonly = False
        self.eval_map = False
        self.eval_map_file = ""
        self.user_grid = False
        self.flipll = False
        self.ext_numvals = 0

        self.nodata = False
        self.fillvalue = 1.e20
        self.ncscale = 1.0

        self.sql_tab_name = ""
        self.fit_file = ""
        self.fit_results_file = 'point'
        self.nc_loaded = False
        self.nc_file_list = 'ncfiles.list'
        self.identical_grid = True
        self.num_ncfiles = 0
        self.nc_files = []
        self.valid_nc = []
        self.nc_jday = []
        self.nc_grid = {}

        self.points_loaded = False
        self.point_file_list = 'points.list'
        self.num_points = 0
        self.point_names = []
        self.pt_lat = []
        self.pt_lon = []
        self.pt_vals = []

        self.rp_ag_type = self.RP_AGTYPE_NONE
        self.rp_ag_mode = self.RP_AGMODE_NONE
        self.base_year = 0
        self.end_year = 0
        self.target_month = 1
        self.rp_ag_interval = 0
        self.index_start = 0
        self.index_end = 0

        self.out_base = ""
        self.nc_varname = ""
        self.nc_varunits = ""
        self.rp_outvals = []
        self.anmax = []
        self.rawdata = []

        self.rpmap_list = 'rp.list'
        self.num_rpmaps = 0
        self.return_periods = []
        self.rp_names = []
        self.rp_units = []

        self.map_grid = {}

        self.weibull_only = False
        self.optimize_weib = False
        self.map_plimits = False
        self.weib_pref = True

        self.mask = False
        self.compute_aal = False
        self.aal = 0
        self.annual_subset = False

        self.weib_fit = FitType()
        self.lognorm_fit = FitType()
        self.ev1max_fit = FitType()
        self.ev1min_fit = FitType()
        self.map_fit = []

        self.weib_new = NewFitType()
        self.lognorm_new = NewFitType()

        self.interval_threshold = 0
        self.master_threshold = 0

        self.nc_nlats = 0
        self.nc_nlons = 0
        self.nc_vals = []
        self.ra_vals = []
        self.nc_lats = []
        self.nc_lons = []

        self.yrchunksizes = [0, 0, 0]
        self.yrdimids = [0, 0, 0]
        self.yr_varid = 0
        self.yrncid = 0
        self.rpncid = 0
        self.rpdimids = [0, 0]
        self.rp_varid = []

        self.spi_mode = False
        self.spi_ncfile = ""
        self.spi_query_value = 0
        self.spi_yy = 0
        self.spi_mm = 0
        self.spi_refyy = 2012
        self.spi_refmm = 12

        self.max_fit_vals = 50
        self.march_size = 10000

        self.nump = 0
        self.tfield = 0
        self.alphap = []
        self.betap = []

        self.fit_server = ""
        self.fit_db = ""
        self.fit_table = ""
        self.fit_week = ""
        self.fit_var = ""
        self.fit_method = ""
        self.fit_id = ""
        self.fit_param1 = ""
        self.fit_param2 = ""
        self.fit_zerofrac = ""
        self.fit_sdp1 = ""
        self.fit_sdp2 = ""
        self.fit_correl = ""
        self.fit_chisq = ""
        self.sql_command = ""
        self.q = ""
        self.tvarnam = ""

    def get_nc_latlon(self, fname, var, lat, lon):
        # Placeholder function; implement netCDF read logic
        rval = 0.0
        return rval

    def generate_weibull_std_plims(self, rp):
        # Placeholder function; implement Weibull standard prediction limits logic
        mle = p10 = p25 = p33 = p50 = p67 = p75 = p90 = 0.0
        p01 = p05 = p95 = p99 = None
        return mle, p10, p25, p33, p50, p67, p75, p90, p01, p05, p95, p99

# Example usage
rpmap = RPMapModule()
print(rpmap.debug)