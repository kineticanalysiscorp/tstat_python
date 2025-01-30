#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 14:41:16 2025

@author: agastraa
"""

import os
import numpy as np
import pandas as pd
import psycopg2

# Weibull distribution function
def weibd(param1, param2, tval):
    return (param1 / param2) * (tval / param2) ** (param1 - 1) * np.exp(-(tval / param2) ** param1)

# Subroutine: solve_external_data
def solve_external_data(fit_file, weib_fit):
    print(f'Analyzing external data file {fit_file}')
    if not os.path.exists(fit_file):
        print(f'EPIC FAIL: {fit_file} does not exist.')
        return

    df = pd.read_csv(fit_file)
    header = df.columns.tolist()
    tfield = 'target_field'  # Replace 'target_field' with actual target field name

    print(f'{len(header)} fields in input file, target {tfield}')

    tvals = df[tfield].to_numpy()
    weibd_values = weibd(weib_fit['param1'], weib_fit['param2'], tvals)

    mlepvals = weibd_values * (1.0 - weib_fit['abs_zero_frac']) + weib_fit['abs_zero_frac']
    mlerps = 1.0 / (1.0 - (weibd_values * (1.0 - weib_fit['abs_zero_frac'])))

    df['mlepct'] = mlepvals
    df['mlerpd'] = mlerps

    output_fname = 'solver_output.csv'
    df.to_csv(output_fname, index=False)

# Subroutine: solve_external_data_sql
def solve_external_data_sql(fit_file, tvarnam, fit_server='cortex1', fit_db='mpas_60km_climo'):
    print(f'Analyzing external data file using 60km MPAS climatology {fit_file}')
    if not os.path.exists(fit_file):
        print(f'EPIC FAIL: {fit_file} does not exist.')
        return

    conn = psycopg2.connect(host=fit_server, dbname=fit_db)
    cursor = conn.cursor()

    df = pd.read_csv(fit_file)
    header = df.columns.tolist()
    if tvarnam not in header or 'grid_id' not in header:
        print('bad target variable name in solve_external_data_sql')
        return

    tvals = df[tvarnam].to_numpy()
    grid_ids = df['grid_id'].to_numpy()

    mlepvals = []
    for grid_id, tval in zip(grid_ids, tvals):
        cursor.execute("SELECT * FROM qc_params WHERE week = 0 AND var = %s AND grid_id = %s;", (tvarnam, grid_id))
        result = cursor.fetchone()
        if result:
            param1, param2, abs_zero_frac = result[4], result[5], result[10]
            mlepval = weibd(param1, param2, tval) * (1.0 - abs_zero_frac) + abs_zero_frac
            mlepvals.append((grid_id, tval, mlepval))

    cursor.close()
    conn.close()

    output_df = pd.DataFrame(mlepvals, columns=['grid_id', 'value', 'pmle'])
    output_df.to_csv('solver_output.csv', index=False)

# Subroutine: gen_weib_plim_params
def gen_weib_plim_params(weib_fit, nump=10000):
    al = weib_fit['param1']
    be = weib_fit['param2']
    sdal = weib_fit['sd_param1']
    sdbe = weib_fit['sd_param2']
    co = weib_fit['correl']

    alphap = np.zeros(nump)
    betap = np.zeros(nump)

    for i in range(nump):
        z1 = np.random.normal()
        z2 = np.random.normal()
        x1 = z1
        x2 = co * z1 + np.sqrt(1.0 - co ** 2) * z2
        alphap[i] = sdal * x1 + al
        betap[i] = sdbe * x2 + be

    return alphap, betap

# Example usage for solve_external_data
weib_fit = {
    'param1': 0.5,
    'param2': 1.2,
    'abs_zero_frac': 0.1
}
solve_external_data('path_to_fit_file.csv', weib_fit)

# Example usage for solve_external_data_sql
solve_external_data_sql('path_to_fit_file.csv', 'target_variable')

# Example usage for gen_weib_plim_params
weib_fit = {
    'param1': 0.5,
    'param2': 1.2,
    'sd_param1': 0.1,
    'sd_param2': 0.2,
    'correl': 0.3
}
alphap, betap = gen_weib_plim_params(weib_fit)