#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 16:29:52 2022

@author: ghiggi
"""
import re
import pandas as pd
import xarray as xr
import numpy as np
from xarray.core import dtypes


def xr_regularize_time_dimension(ds: xr.Dataset,
                                 t_res: str = "2min30s",
                                 fill_value: float = np.nan):
    """Fill missing timesteps with default values in a xarray Dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Input xarray Dataset
    range_freq : str, optional
        Interval between each timestep, by default "2min30s"
    fill_value : optional
        Fill value for missing timesteps

    Returns
    -------
    xr.Dataset
        Dataset with a contiguous time range and no missing timesteps
    """
    start = ds.time.to_numpy()[0]
    end = ds.time.to_numpy()[-1]
    start_date = pd.to_datetime(start).date()
    end_date = pd.to_datetime(pd.to_datetime(end).date()) + \
        pd.Timedelta(hours=23, minutes=57, seconds=30)
    full_range = pd.date_range(start=start_date,
                               end=end_date,
                               freq=t_res).to_numpy()
    ds_reindexed = ds.reindex({"time": full_range}, fill_value=fill_value)
    return ds_reindexed


def xr_keep_quality_radar_measurements(ds: xr.Dataset):
    """Drop duplicate timesteps in the input Dataset by only keeping the
    data measurements taken with the highest number of available radars.

    Parameters
    ----------
    ds : xr.Dataset
        Input xarray Dataset

    Returns
    -------
    xr.Dataset
        Processed dataset with no duplicate timesteps
    """
    idx_keep = np.arange(len(ds.time))
    to_remove = []
    _, idx, count = np.unique(ds.time, return_counts=True, return_index=True)
    index_duplicates = [list(range(idx[i], idx[i]+count[i])) for i, _ in
                        enumerate(idx) if count[i] > 1]

    for dup in index_duplicates:
        radar_names = [len(re.sub(r'[^a-zA-Z]', '',
                                  str(ds.radar_names[i].values))) for i in dup]
        to_remove.extend([dup[i] for i in range(len(radar_names)) if
                          i != radar_names.index(max(radar_names))])

    idx_keep = [i for i in idx_keep if i not in to_remove]
    ds = ds.isel(time=idx_keep)
    return ds


def ensure_regular_timesteps(ds: xr.Dataset,
                             fill_value: dict = {"precip": dtypes.NA, "mask": 0},
                             t_res: str = "2min30s"):
    ds = xr_keep_quality_radar_measurements(ds.sortby("time"))
    ds = xr_regularize_time_dimension(ds,
                                      t_res=t_res,
                                      fill_value=fill_value)
    return ds