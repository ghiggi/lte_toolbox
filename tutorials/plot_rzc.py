#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 15:27:18 2022

@author: ghiggi
"""
import os 
import zarr 
import xarray as xr 
import lte_mch_toolbox
from lte_toolbox.mch.visualization.plot import plot_precip_field
from lte_toolbox.mch.visualization.plot import plot_precip_fields

rzc_zarr_fpath = os.path.join("/ltenas3/0_Data/NowProject/", "zarr", "rzc_temporal_chunk.zarr")
# TODO: 
# rzc_zarr_fpath = os.path.join("/ltenas3/0_MCH/RZC/zarr", "rzc_temporal_chunk.zarr")

ds_rzc = xr.open_zarr(rzc_zarr_fpath)
ds_rzc.attrs

# TODO temporary 
ds_rzc = ds_rzc.assign_coords({"x": ds_rzc["x"].data*1000})
ds_rzc = ds_rzc.assign_coords({"y": ds_rzc["y"].data*1000})



da = ds_rzc['precip'].isel(time=0)
plot_precip_field(da)

da = ds_rzc['precip'].isel(time=slice(0,4))
plot_precip_fields(da, col="time", col_wrap="2")
