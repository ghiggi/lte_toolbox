#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 15:35:15 2022

@author: ghiggi
"""

# --> decoded = scale_factor * encoded + add_offset
# --> encoded = (decoded - add_offset)/ scale_factor
DATA_ENCODING_PER_PID = {
    "AZC": {
        # rainfall accumulation, mm
        # Range from 10^-2 to 10^1
        'dtype': "uint16",
        '_FillValue': 65535,
        "scale_factor": 0.01,
        "add_offset": 0.0
    },
    "BZC": {
        # probability of hail, percentage
        # Range from 0 to 100
        'dtype': "uint8",
        '_FillValue': 255,
        "add_offset": 0.0
    },
    "CPC": {
        # combiprecip, mm
        # 10^-3 to 10^1
        'dtype': "uint16",
        '_FillValue': 65535,
        "scale_factor": 0.001,
        "add_offset": 0.0
    },
    "CZC": {
        # max column height across sweeps, dBZ
        'dtype': "uint8",
        '_FillValue': 255,
        "add_offset": -40
    },
    "EZC": {
        # Echotop 15, km
        # 0, 1.4 to 50-60
        'dtype': "uint16",
        '_FillValue': 65535,
        "scale_factor": 0.1,
        "add_offset": 0.0
    },
    "LZC": {
        # VIL, kg/m2
        # 0, 0.5, 1 ..., 30...
        'dtype': "uint16",
        '_FillValue': 65535,
        "scale_factor": 0.1,
        "add_offset": 0.0
    },
    "MZC": {
        # MESH, mm
        # float: 0, 2., 2.1 .., 4.5..
        'dtype': "uint8",
        '_FillValue': 255,
        "scale_factor": 0.1,
        "add_offset": 0.0
    },
    "HZC": {

    },
    "NHC": {

    },
    "NZC": {

    },
    "OZC": {

    },
    "RZC": {
        'dtype': "uint16",
        '_FillValue': 65535,
        "scale_factor": 0.01,
        "add_offset": 0.0
    },
}


mask_netcdf_encoding = {
    'zlib': True,
    'shuffle': True,
    'complevel': 1,
    'fletcher32': False,
    'contiguous': False,
    'chunksizes': (24, 128, 142),
    'dtype': 'uint8',
    "_FillValue": 0,
}

data_netcdf_encoding = {
    'zlib': True,
    'shuffle': True,
    'complevel': 1,
    'fletcher32': False,
    'contiguous': False,
    'chunksizes': (24, 128, 142),
}

MCH_NETCDF_ENCODINGS = {
    "mask": mask_netcdf_encoding,
    "data": data_netcdf_encoding
}

mask_zarr_encoding = {
    "dtype": "uint8",
    "_FillValue": 0,
}

MCH_ZARR_ENCODINGS = {
    "mask": mask_zarr_encoding
}
