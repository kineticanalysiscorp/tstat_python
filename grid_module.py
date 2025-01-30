#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 12:40:07 2025

@author: agastraa
"""

import numpy as np

class GridSpec:
    def __init__(self, xspan, yspan, xmin, ymin, xmax, ymax, xres, yres, reversed=False, gmtref=False):
        self.xspan = xspan
        self.yspan = yspan
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.xres = xres
        self.yres = yres
        self.reversed = reversed
        self.gmtref = gmtref

def grids_equiv(g1, g2):
    """
    Compare two grid specifications to determine if they are equivalent.
    
    Parameters:
    g1 (GridSpec): First grid specification.
    g2 (GridSpec): Second grid specification.
    
    Returns:
    bool: True if the grid specifications are equivalent, False otherwise.
    """
    return (g1.xspan == g2.xspan and
            g1.yspan == g2.yspan and
            g1.xmin == g2.xmin and
            g1.ymin == g2.ymin and
            g1.xmax == g2.xmax and
            g1.ymax == g2.ymax)

def ll_to_gridxy(gspec, lat, lon):
    """
    Convert latitude and longitude to grid x and y coordinates.
    
    Parameters:
    gspec (GridSpec): Grid specification.
    lat (float): Latitude.
    lon (float): Longitude.
    
    Returns:
    tuple: (xo, yo, ongrid)
        xo (int): Grid x coordinate.
        yo (int): Grid y coordinate.
        ongrid (bool): True if the coordinates are on the grid, False otherwise.
    """
    glon = lon
    if gspec.gmtref and glon < 0.0:
        glon += 360.0
    
    x = (glon - gspec.xmin) / gspec.xres
    y = (lat - gspec.ymin) / gspec.yres

    if gspec.reversed:
        y = (gspec.ymax - lat) / gspec.yres

    xo = int(np.floor(x + 0.5)) + 1
    yo = int(np.floor(y + 0.5)) + 1

    ongrid = True
    if xo < 1:
        ongrid = False
        xo = 1
    if yo < 1:
        ongrid = False
        yo = 1
    if xo > gspec.xspan:
        ongrid = False
        xo = gspec.xspan
    if yo > gspec.yspan:
        ongrid = False
        yo = gspec.yspan

    return xo, yo, ongrid

def gridxy_to_ll(gspec, x, y):
    """
    Convert grid x and y coordinates to latitude and longitude.
    
    Parameters:
    gspec (GridSpec): Grid specification.
    x (int): Grid x coordinate.
    y (int): Grid y coordinate.
    
    Returns:
    tuple: (lat, lon)
        lat (float): Latitude.
        lon (float): Longitude.
    """
    xp = x - 1.0
    yp = y - 1.0

    lat = gspec.ymin + (yp) * gspec.yres
    if gspec.reversed:
        lat = gspec.ymax - (yp) * gspec.yres

    lon = gspec.xmin + (xp) * gspec.xres
    if gspec.gmtref:
        lon = gspec.xmin + (xp) * gspec.xres

    return lat, lon

# Example usage
gspec = GridSpec(100, 100, 0, 0, 10, 10, 0.1, 0.1)
print(grids_equiv(gspec, gspec))  # Should print: True
print(ll_to_gridxy(gspec, 5, 5))  # Example conversion
print(gridxy_to_ll(gspec, 50, 50))  # Example conversion