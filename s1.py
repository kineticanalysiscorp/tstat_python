#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 13:31:52 2025

@author: agastraa
"""

import os
import numpy as np
import pandas as pd

def weibd(param1, param2, tval):
    """
    Weibull distribution function.
    Replace this with the actual implementation if needed.
    """
    # Placeholder for the actual Weibull distribution calculation
    return (param1 / param2) * (tval / param2) ** (param1 - 1) * np.exp(-(tval / param2) ** param1)

def solve_external_data(fit_file, weib_fit):
    """
    Analyze external data file and compute mlepct and mlerpd values.
    """
    print(f'Analyzing external data file {fit_file}')
    if not os.path.exists(fit_file):
        print(f'EPIC FAIL: {fit_file} does not exist.')
        return

    # Read the CSV file into a DataFrame
    df = pd.read_csv(fit_file)
    header = df.columns.tolist()
    if 'target_field' not in header:
        print('Target field not found in the file header.')
        return
    
    # Define the target field
    tfield = 'target_field'  # Replace 'target_field' with actual target field name

    print(f'{len(header)} fields in input file, target {tfield}')

    # Vectorized operation for weibd
    tvals = df[tfield].to_numpy()
    weibd_values = weibd(weib_fit['param1'], weib_fit['param2'], tvals)
    
    mlepvals = weibd_values * (1.0 - weib_fit['abs_zero_frac']) + weib_fit['abs_zero_frac']
    mlerps = 1.0 / (1.0 - (weibd_values * (1.0 - weib_fit['abs_zero_frac'])))

    # Append new columns to DataFrame
    df['mlepct'] = mlepvals
    df['mlerpd'] = mlerps

    # Write the output to a new CSV file
    output_fname = 'solver_output.csv'
    df.to_csv(output_fname, index=False)

# Example usage
weib_fit = {
    'param1': 0.5,
    'param2': 1.2,
    'abs_zero_frac': 0.1
}
solve_external_data('path_to_fit_file.csv', weib_fit)