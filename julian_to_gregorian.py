#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 12:29:57 2025

@author: agastraa
"""

import jdcal

def caldat(tjd):
    """
    Convert Julian Date to Gregorian calendar date using jdcal module.
    
    Parameters:
    tjd (float): Julian Date
    
    Returns:
    tuple: (year, month, day, hours)
        year (int): Year
        month (int): Month
        day (int): Day
        hours (float): UT hours
    """
    # jdcal.jd2gcal returns a list [year, month, day, fraction_of_day]
    cal_date = jdcal.jd2gcal(tjd, 0)
    year, month, day = cal_date[:3]
    fraction_of_day = cal_date[3]
    hours = fraction_of_day * 24

    return year, month, day, hours

