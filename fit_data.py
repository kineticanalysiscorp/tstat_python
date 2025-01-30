#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 17:32:24 2025

@author: agastraa
"""

import numpy as np
from scipy.stats import weibull_min, lognorm

class FitData:
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
        self.have_ovals = False
        self.ovals = None
        self.nfit = 0
        self.thres_rval = 0
        self.thres_pval = 0
        self.nzero = 0
        self.nvals = 0

# Initialize global variables
weib_new = FitData()
weib_fit = FitData()
lognorm_fit = FitData()
ev1max_fit = FitData()
ev1min_fit = FitData()
max_fit_vals = 50
debug = False
master_threshold = 0
optimize_weib = False

def weibull_fit(data):
    # Actual Weibull fitting logic using scipy
    c, loc, scale = weibull_min.fit(data, floc=0)
    chisq = np.sum((data - weibull_min(c, loc, scale).cdf(data)) ** 2)
    return c, scale, 0, 0, 0, chisq

def fit_data_new(rvals, oval_flag=False):
    nrvals = len(rvals)
    xord = np.sort(rvals)
    nzero = np.sum(xord > 0)

    weib_new.have_ovals = oval_flag

    if nzero > max_fit_vals:
        nfit = max_fit_vals
        thres = xord[-nfit]
    else:
        nfit = nzero
        thres = 0.0

    tail = xord[-nfit:] - thres
    x = tail
    numxi = nfit
    delbin = (x[-1] - x[1]) / 10.0  # Assuming numbins = 10

    # Weibull fitting
    weib_new.param1, weib_new.param2, weib_new.sd_param1, weib_new.sd_param2, weib_new.correl, weib_new.chisq = weibull_fit(x)

    if debug:
        print('New Weibull fit chisq:', weib_new.chisq)
        print('New Weibull fit Params a b:', weib_new.param1, weib_new.param2)
        print('New Weibull fit Params sd:', weib_new.sd_param1, weib_new.sd_param2, weib_new.correl)

    if weib_new.have_ovals:
        weib_new.ovals = xord

    weib_new.minval = np.min(xord)
    weib_new.nfit = nfit
    weib_new.thres_rval = thres
    weib_new.thres_pval = (nrvals - nfit) * 1.0 / nrvals
    weib_new.nzero = nzero
    weib_new.nvals = nrvals

def get_weib_rp(y):
    flag = True
    if y > weib_new.thres_rval:
        yp = y - weib_new.thres_rval
        rval = weibull_min.cdf(yp, weib_new.param1, scale=weib_new.param2)
        rp = weib_new.thres_pval + rval * (1.0 - weib_new.thres_pval)
    else:
        flag = False
        ix = np.searchsorted(weib_new.ovals, y)
        rp = ix * 1.0 / weib_new.nvals
    return rp, flag

def get_weib_val(rp):
    flag = True
    if rp > weib_new.thres_pval:
        rval = (rp - weib_new.thres_pval) / (1.0 - weib_new.thres_pval)
        y = weibull_min.ppf(rval, weib_new.param1, scale=weib_new.param2) + weib_new.thres_rval
    else:
        flag = False
        rval = rp * weib_new.nvals
        if rval < 1:
            rval = 1
        if weib_new.have_ovals:
            y = weib_new.ovals[int(rval) - 1]
        else:
            y = weib_new.minval + (weib_new.thres_rval - weib_new.minval) * rval
    return y, flag

def get_all_rp_val(rp):
    zerofrac = weib_fit.abs_zero_frac
    thres = weib_fit.thres_used

    if rp > zerofrac:
        rpval = (rp - zerofrac) / (1.0 - zerofrac)
        wb = weibull_min.ppf(rpval, weib_fit.param1, scale=weib_fit.param2) + thres
        lno = lognorm.ppf(rpval, lognorm_fit.param1, scale=lognorm_fit.param2) + thres
        ev1l = evtype1pp(rpval, ev1min_fit.param1, ev1min_fit.param2) + thres
        ev1u = evtype1pp(rpval, ev1max_fit.param1, ev1max_fit.param2) + thres
    else:
        wb = lno = ev1l = ev1u = 0.0
    return wb, lno, ev1l, ev1u

def evtype1pp(rpval, param1, param2):
    # Placeholder: replace with actual EV type 1 fitting logic
    return rpval * param1 + param2

def fit_data(rvals):
    nrvals = len(rvals)
    z = np.sort(rvals)
    thres = 0.001
    if nrvals > max_fit_vals:
        thres = z[-max_fit_vals]
    if master_threshold != 0:
        thres = master_threshold

    weib_fit.valid = False
    lognorm_fit.valid = False

    numxi = np.sum(z > thres)
    weib_fit.abs_zero_frac = np.sum(z <= 0) / nrvals

    zerofrac = 1.0 - numxi / nrvals
    mean = zerofrac * nrvals
    variance = nrvals * zerofrac * (1.0 - zerofrac)
    stdev = np.sqrt(variance)

    weib_fit.zf_nvals = nrvals
    weib_fit.zf_mean = mean
    weib_fit.zf_stdev = stdev

    if debug:
        print('Threshold fraction:', zerofrac, 'using', thres)
        print('  Mean ZF:', weib_fit.zf_nvals)
        print('  StdDev:', weib_fit.zf_stdev)

    y = z
    numxi = np.sum(y > thres)
    x = y[y > thres] - thres

    if numxi < 5:
        return

    weib_fit.thres_used = thres

    delbin = (x[-1] - x[0]) / 10.0  # Assuming numbins = 10
    i = int(0.5 * nrvals)
    if i > (nrvals - numxi):
        i = nrvals - numxi
    if i < 1:
        i = 1

    weib_fit.minval = z[i - 1]
    weib_fit.param1, weib_fit.param2, weib_fit.sd_param1, weib_fit.sd_param2, weib_fit.correl, weib_fit.chisq = weibull_fit(x)

    if debug:
        print('Weibull fit chisq:', weib_fit.chisq)
        print('Weibull fit Params:', weib_fit.param1, weib_fit.param2)
        print('Weibull fit minval:', weib_fit.minval)

    weib_fit.valid = True

    # Lognormal fitting (replace this with actual fitting logic)
    lognorm_fit.param1, lognorm_fit.param2, lognorm_fit.sd_param1, lognorm_fit.sd_param2, lognorm_fit.correl, lognorm_fit.chisq = lognorm_fit_func(x)

    if debug:
        print('Log Normal Fit chisq:', lognorm_fit.chisq)

    if optimize_weib:
        optimize_weibull(z, nrvals, zerofrac)

def lognorm_fit_func(data):
    # Placeholder: replace with actual Log Normal fitting logic
    s, loc, scale = lognorm.fit(data, floc=0)
    chisq = np.sum((data - lognorm(s, loc, scale).cdf(data)) ** 2)
    return s, scale, 0, 0, 0, chisq

def optimize_weibull(z, nrvals, zerofrac):
    del_weib = del_empr = 0
    for i in range(nrvals - 1):
        rp = (i * 1.0) / nrvals
        if rp > zerofrac:
            rpval = (rp - zerofrac) / (1.0 - zerofrac)
            rweib = weibull_min.ppf(rpval, weib_fit.param1, scale=weib_fit.param2) + weib_fit.thres_used
            if i >= nrvals - 5:
                del_weib += rweib
                del_empr += z[int(rp * nrvals) + 1]

    del_weib /= 4.0
    del_empr /= 4.0
    p2est = weib_fit.param2 * del_empr / del_weib

    if not np.isnan(p2est):
        if weib_fit.param2 < p2est:
            weib_fit.param2 = p2est
    else:
        if debug:
            print('weib optimization failed')
            

def generate_weibull_mle(data):
    """
    Fit data to a Weibull distribution using Maximum Likelihood Estimation (MLE).

    Parameters:
    data (numpy.ndarray): Array of data points.

    Returns:
    tuple: (shape, scale)
        shape (float): Shape parameter of the Weibull distribution.
        scale (float): Scale parameter of the Weibull distribution.
    """
    n = len(data)
    log_sum = np.sum(np.log(data))
    inv_sum = np.sum(1.0 / data)
    sum_log_sum = np.sum(np.log(data) ** 2)

    mean_log = log_sum / n
    mean_inv = inv_sum / n

    shape = (mean_log - np.log(mean_inv)) / (sum_log_sum / n - mean_log ** 2)
    scale = mean_inv / np.exp(mean_log / shape)

    return shape, scale

# Example usage:
data = np.array([1.2, 2.3, 3.4, 4.5, 5.6])
shape, scale = generate_weibull_mle(data)
print(f"Shape: {shape}, Scale: {scale}")

            

def generate_weibull_std_plims(rp):
    mle = p10 = p25 = p33 = p50 = p67 = p75 = p90 = 0.0
    p01 = p05 = p95 = p99 = None

    zerofrac = weib_fit.zf_mean / weib_fit.zf_nvals
    if rp > zerofrac:
        rpval = (rp - zerofrac) / (1.0 - zerofrac)
        mle = weibull_min.ppf(rpval, weib_fit.param1, scale=weib_fit.param2) + weib_fit.thres_used
    else:
        mle = 0.0
        if zerofrac > weib_fit.abs_zero_frac and rp > weib_fit.abs_zero_frac:
            rpy1 = (rp - 0.5) / 0.5
            if rpy1 < 0:
                rpy1 = 0
            mle = weib_fit.minval + (weib_fit.thres_used - weib_fit.minval) * rpy1 * rpy1

    nr = 10000
    sdvals = np.zeros(nr)
    for i in range(nr):
        zerofrac = (weib_fit.zf_mean + weib_fit.zf_stdev * np.random.randn()) / weib_fit.zf_nvals
        if rp > zerofrac:
            rpval = (rp - zerofrac) / (1.0 - zerofrac)
            z1 = np.random.randn()
            z2 = np.random.randn()
            x1 = z1
            x2 = weib_fit.correl * z1 + np.sqrt(1.0 - weib_fit.correl ** 2) * z2
            a1 = weib_fit.sd_param1 * x1 + weib_fit.param1
            b1 = weib_fit.sd_param2 * x2 + weib_fit.param2
            sdvals[i] = weibull_min.ppf(rpval, a1, scale=b1) + weib_fit.thres_used
        else:
            sdvals[i] = 0.0
            if zerofrac > weib_fit.abs_zero_frac and rp > weib_fit.abs_zero_frac:
                rpy1 = (rp - 0.5) / 0.5
                if rpy1 < 0:
                    rpy1 = 0
                sdvals[i] = weib_fit.minval + (weib_fit.thres_used - weib_fit.minval) * rpy1 * rpy1

    sdvals.sort()
    p01 = sdvals[int(nr * 0.01)]
    p05 = sdvals[int(nr * 0.05)]
    p10 = sdvals[int(nr * 0.10)]
    p25 = sdvals[int(nr * 0.25)]
    p33 = sdvals[int(nr * 0.33333)]
    p50 = sdvals[int(nr * 0.50)]
    p67 = sdvals[int(nr * 0.66667)]
    p75 = sdvals[int(nr * 0.75)]
    p90 = sdvals[int(nr * 0.90)]
    p95 = sdvals[int(nr * 0.95)]
    p99 = sdvals[int(nr * 0.99)]

    return mle, p10, p25, p33, p50, p67, p75, p90, p01, p05, p95, p99

# Example usage
rvals = np.random.weibull(2, 1000)
fit_data_new(rvals)
print(get_weib_rp(1.5))
print(get_weib_val(0.95))
print(generate_weibull_std_plims(0.95))