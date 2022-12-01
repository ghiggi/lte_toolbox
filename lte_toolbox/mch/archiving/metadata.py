#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 15:34:03 2022

@author: ghiggi
"""
import pyproj

crs = pyproj.CRS.from_epsg(21781)


METADATA = {
    "EPSG":  21781,
    "projection": crs.to_proj4(),
    "PROJ_parameters": crs.to_json(),
    "x1": 255000.0,
    "y1": -160000.0,
    "x2": 965000.0,
    "y2": 480000.0,
    "xpixelsize": 1000.0,
    "ypixelsize": 1000.0,
    "cartesian_unit": "m",
    "yorigin": "upper",
    "institution": "MeteoSwiss",
}

METADATA_CH = {
    "EPSG":  21781,
    "projection": crs.to_proj4(),
    "PROJ_parameters": crs.to_json(),
    "x1": 486000.0,
    "y1": 76000.0,
    "x2": 831000.0,
    "y2": 301000.0,
    "xpixelsize": 1000.0,
    "ypixelsize": 1000.0,
    "cartesian_unit": "m",
    "yorigin": "upper",
    "institution": "MeteoSwiss",
}