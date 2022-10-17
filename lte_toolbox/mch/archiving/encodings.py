#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 15:35:15 2022

@author: ghiggi
"""


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


mask_zarr_encoding = {
    "dtype": "uint8",
    "_FillValue": 0,
}


precip_netcdf_encoding = {
    'zlib': True,
    'shuffle': True,
    'complevel': 1,
    'fletcher32': False,
    'contiguous': False,
    'chunksizes': (24, 128, 142),
    'dtype': "uint16",
    '_FillValue': 65535,
    "scale_factor": 0.01,
    "add_offset": 0.0
 }


precip_zarr_encoding = { 
    "dtype": "uint16", 
    '_FillValue': 65535,
    "scale_factor": 0.01,
    "add_offset": 0.0,
}


RZC_NETCDF_ENCODINGS = {
    "mask": mask_netcdf_encoding,
    "precip": precip_netcdf_encoding
}


RZC_ZARR_ENCODINGS = {
    'precip': precip_zarr_encoding,
    "mask": mask_zarr_encoding
}


