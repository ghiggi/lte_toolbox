#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 15:40:05 2022

@author: ghiggi
"""
import os 
import zarr 
import pandas as pd
import xarray as xr 
import matplotlib.pyplot as plt 
from PIL import Image
from lte_toolbox.mch.visualization import plot_precip_field

rzc_zarr_fpath = os.path.join("/ltenas3/0_Data/NowProject/", "zarr", "rzc_temporal_chunk.zarr")
figs_dir = "/tmp"
# TODO: 
# rzc_zarr_fpath = os.path.join("/ltenas3/0_Data/MCH/RZC/zarr", "rzc_temporal_chunk.zarr")

ds_rzc = xr.open_zarr(rzc_zarr_fpath)
ds_rzc.attrs

# TODO temporary 
ds_rzc = ds_rzc.assign_coords({"x": ds_rzc["x"].data*1000})
ds_rzc = ds_rzc.assign_coords({"y": ds_rzc["y"].data*1000})

##---------------------------------------------------------------------.
# Create figure temporary directory 
tmp_figs_dir = os.path.join(figs_dir, "tmp")
os.makedirs(figs_dir, exist_ok=True)
os.makedirs(tmp_figs_dir, exist_ok=True)
 
# Select DataArray
da = ds_rzc["precip"]

# Inialize list of frames 
list_frames = []

# Create the video/gif frames 
for i, time in enumerate(da.time.values):
    # Define frame fpath 
    time_str = str(time.astype('datetime64[s]'))
    filepath = os.path.join(tmp_figs_dir, f"{time_str}.png")
     
    # Select single frame 
    tmp_da = da.isel(time=i)
    
    # Plot and save the frame 
    figsize = (8, 5)
    title = "RZC, Time: {}".format(time_str)
    ax, p = plot_precip_field(tmp_da, title=title, figsize=figsize)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    
    # Append to list 
    list_frames.append(Image.open(filepath).convert("P",palette=Image.ADAPTIVE))
 
##---------------------------------------------------------------------.
# Create the GIF 
fps = 4 
date = str(pd.to_datetime(da.time.values[0]).date())
gif_fpath = os.path.join(figs_dir, f"{date}.gif")
list_frames[0].save(
    gif_fpath,
    format="gif",
    save_all=True,
    append_images=list_frames[1:],
    duration=1 / fps * 1000,  # ms
    loop=False,
)

##---------------------------------------------------------------------.
