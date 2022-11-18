#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 15:14:44 2022

@author: ghiggi
"""
import shutil
import pathlib
import datetime

from time import time

# import pyart
import zarr
import pandas as pd
import numpy as np
import xarray as xr
from xarray.core import dtypes

from typing import Any
from zipfile import ZipFile
from itertools import repeat
from multiprocessing import Pool

from lte_toolbox.mch.archiving.encodings import (
    RZC_NETCDF_ENCODINGS,
    RZC_ZARR_ENCODINGS)
from lte_toolbox.mch.archiving.metadata import RZC_METADATA
from lte_toolbox.mch.archiving.utils import xr_keep_quality_radar_measurements
from lte_toolbox.mch.archiving.utils import xr_regularize_time_dimension

from xforecasting.utils.zarr import write_zarr

RZC_BOTTOM_LEFT_COORDINATES = [255, -160]

# -----------------------------------------------------------------------------.
# Extraction


def get_time_from_rzc_filename(filename: str) -> datetime.datetime:
    """Determines the time corresponding to the RZC filename.

    Parameters
    ----------
    filename : str
        Name of the RZC file.

    Returns
    -------
    datetime.datetime
        Time corresponding to the RZC filename
    """
    time = datetime.datetime.strptime(filename[3:12], "%y%j%H%M")
    if filename[3:12].endswith("2") or filename[3:12].endswith("7"):
        time = time + datetime.timedelta(seconds=30)
    return time


def get_coords_from_corners(rzc):
    rzc_shape = rzc.shape
    x = np.arange(RZC_BOTTOM_LEFT_COORDINATES[0],
                  RZC_BOTTOM_LEFT_COORDINATES[0] + rzc_shape[1]) + 0.5
    y = np.arange(RZC_BOTTOM_LEFT_COORDINATES[1] + rzc_shape[0] - 1,
                  RZC_BOTTOM_LEFT_COORDINATES[1] - 1, -1) + 0.5
    x = x*1000
    y = y*1000

    return x, y


def read_rzc_file(input_path: pathlib.Path, reader="python") -> xr.Dataset:
    """Reads the RZC file and returns an xarray Dataset containing
    the precipitation estimates with the metadata of the file.

    Parameters
    ----------
    input_path : pathlib.Path
        Path to input RZC file

    Returns
    -------
    xr.Dataset
        Dataset containing the precipitation estimates along with
        the metadata of the file
    """
    # Get cartesian metranet class instance
    metranet = pyart.aux_io.read_cartesian_metranet(input_path.as_posix(),
                                                    reader=reader)
    # Get RZC array
    rzc = metranet.fields['radar_estimated_rain_rate']['data'][0, :, :]
    # Get RZC metadata
    metranet_header = get_metranet_header_dictionary(input_path.as_posix())
    # Compute CH1903 coordinates (at pixel centroids)
    # - Provide in meters for compatibility with Cartopy and unit of
    # CRS definition
    x, y = get_coords_from_corners(rzc)
    # Get timestep from filename
    time = get_time_from_rzc_filename(input_path.as_posix().split("/")[-1])
    # Get radar availability
    radar_availability = input_path.as_posix().split("/")[-1][12:14]
    # Create xr.Dataset
    ds = xr.Dataset(
                data_vars=dict(
                    precip=(["y", "x"], rzc.data[:, ::-1, :]),
                    mask=(["y", "x"], rzc.mask.astype(int)[:, ::-1, :]),
                ),
                coords=dict(
                    time=time,
                    x=(["x"], x),
                    y=(["y"], y),
                    radar_availability=radar_availability,
                    radar_names=metranet_header.get("radar", ""),
                    radar_quality=metranet_header.get("quality", "")
                ),
                attrs=RZC_METADATA,
            )
    return ds


def daily_rzc_data_to_netcdf(input_dir_path: pathlib.Path,
                             output_dir_path: pathlib.Path,
                             log_dir_path: pathlib.Path,
                             fill_value={"precip": dtypes.NA, "mask": 0},
                             t_res="2min30s",
                             encoding: dict = RZC_NETCDF_ENCODINGS):
    """Read all the RZC files of a certain day, combine them into one xarray
    Dataset and save the result in a netCDF file.

    Parameters
    ----------
    input_dir_path : pathlib.Path
        Path to the input folder
    output_dir_path : pathlib.Path
        Path to the output folder
    log_dir_path : pathlib.Path
        Path to save the logs of the reading process
    encoding : dict, optional
        Encoding of the netCDF file, by default NETCDF_ENCODINGS
    """
    filename = input_dir_path.as_posix().split("/")[-1]
    output_filename = filename + ".nc"
    list_ds = []
    if not (output_dir_path / output_filename).exists():
        for file in sorted(input_dir_path.glob("*.801")):
            try:
                list_ds.append(read_rzc_file(file))
            except TypeError:
                with open(log_dir_path / f"type_errors_{filename}.txt", "a+") as f:
                    f.write(f"{file.as_posix()}\n")
            except ValueError:
                with open(log_dir_path / f"value_errors_{filename}.txt", "a+") as f:
                    f.write(f"{file.as_posix()}\n")

        if len(list_ds) > 0:
            ensure_regular_timesteps(xr.concat(list_ds, dim="time",
                                               coords="all"),
                                     fill_value=fill_value, t_res=t_res)\
                .to_netcdf(output_dir_path / output_filename,
                           encoding=encoding)


def rzc_to_netcdf(data_dir_path: pathlib.Path,
                  output_dir_path: pathlib.Path,
                  log_dir_path: pathlib.Path,
                  num_workers: int = 6):
    """Process all the RZC files and combine them into daily netCDF files.

    Parameters
    ----------
    data_dir_path : pathlib.Path
        Path to the input data folder
    output_dir_path : pathlib.Path
        Path to the output data folder
    log_dir_path : pathlib.Path
        Path to the folder where to save the logs of the
        reading process
    num_workers : int, optional
        Number of workers to parallelize the reading of RZC files, by default 6
    """
    for folder_year in sorted(data_dir_path.glob("*")):
        year = int(folder_year.as_posix().split("/")[-1])
        print(year)
        output_dir_year_path = output_dir_path / str(year)
        output_dir_year_path.mkdir(exist_ok=True)

        list_folders_days = list(sorted(folder_year.glob("*")))
        with Pool(num_workers) as p:
            p.starmap(daily_rzc_data_to_netcdf,
                      zip(list_folders_days, repeat(output_dir_year_path),
                          repeat(log_dir_path)))


# TODO: Refactor on the fly functions
def produce_daily_files(input_path: pathlib.Path,
                        output_zip_path: pathlib.Path,
                        output_netcdf_path: pathlib.Path,
                        log_dir_path: pathlib.Path):
    with ZipFile(input_path, 'r') as zip_ref:
        zip_name = input_path.as_posix().split("/")[-1][:-4]
        output_zip_day_path = output_zip_path / zip_name
        output_zip_day_path.mkdir(exist_ok=True)

        zip_ref.extractall(output_zip_day_path)
        daily_rzc_data_to_netcdf(output_zip_day_path, output_netcdf_path,
                                 log_dir_path)
        shutil.rmtree(output_zip_day_path.as_posix())


def unzip_and_combine_rzc(input_dir_path: pathlib.Path,
                          output_zip_path: pathlib.Path,
                          output_netcdf_path: pathlib.Path,
                          log_dir_path: pathlib.Path,
                          data_start_year: int = 2016,
                          num_workers: int = 2):
    folders = input_dir_path.glob("*")
    for folder in sorted(folders):
        year = int(folder.as_posix().split("/")[-1])
        if year >= data_start_year:
            print(f"{year}.. ")
            output_zip_year_path = output_zip_path / str(year)
            output_netcdf_year_path = output_netcdf_path / str(year)
            output_zip_year_path.mkdir(exist_ok=True)
            output_netcdf_year_path.mkdir(exist_ok=True)
            days = folder.glob("*")
            for day in sorted(days):
                list_folders_days = list(sorted(day.glob("RZC*.zip")))
                with Pool(num_workers) as p:
                    p.starmap(produce_daily_files,
                              zip(list_folders_days,
                                  repeat(output_zip_year_path),
                                  repeat(output_netcdf_year_path),
                                  repeat(log_dir_path)))


def ensure_regular_timesteps(ds: xr.Dataset,
                             fill_value={"precip": dtypes.NA, "mask": 0},
                             t_res="2min30s"):
    ds = xr_keep_quality_radar_measurements(ds.sortby("time"))
    ds = xr_regularize_time_dimension(ds,
                                      t_res=t_res,
                                      fill_value=fill_value)
    return ds


# -----------------------------------------------------------------------------.
# Zarr processing


def netcdf_rzc_to_zarr(data_dir_path: pathlib.Path,
                       output_dir_path: pathlib.Path,
                       compressor: Any = "auto",
                       encoding: dict = RZC_ZARR_ENCODINGS):
    temporal_chunk_filepath = output_dir_path / "chunked_by_time.zarr"
    if temporal_chunk_filepath.exists():
        shutil.rmtree(temporal_chunk_filepath)
    # --------------------------------------------------.
    # Load the dataset
    fpaths = [p.as_posix() for p in
              sorted(list(data_dir_path.glob("*/RZC*.nc")))]
    list_ds = [xr.open_dataset(p, chunks={"time": 576, "x": -1, "y": -1})
               for p in fpaths]
    ds = xr.concat(list_ds, dim="time")
    # ds.attrs = RZC_METADATA # TODO: MAYBE ADD ALREADY TO NETCDF
    # --------------------------------------------------.
    # Write Zarr by block of time
    write_zarr(
        temporal_chunk_filepath.as_posix(),
        ds,
        chunks={"time": 25, "y": -1, "x": -1},
        compressor=compressor,
        rounding=None,
        encoding=encoding,
        consolidated=True,
        append=False,
        show_progress=True,
    )


def load_rzc_zarr(rzc_zarr_filepath: pathlib.Path,
                  mask_to_uint: bool = True) -> xr.Dataset:
    ds = xr.open_zarr(rzc_zarr_filepath)
    ds['radar_quality'] = ds['radar_quality'].astype(str)
    ds['radar_availability'] = ds['radar_availability'].astype(str)
    ds['radar_names'] = ds['radar_names'].astype(str)
    if mask_to_uint:
        ds["mask"] = ds.mask.astype("uint8")
    return ds


def rechunk_zarr_per_pixel(ds: xr.Dataset,
                           output_dir_path: pathlib.Path,
                           chunks: dict,
                           chunk_date_frequency: str = "6MS",
                           compressor: Any = "auto",
                           encoding: dict = RZC_ZARR_ENCODINGS,
                           zarr_filename: str = "rzc_chunked_by_pixel.zarr"):
    spatial_chunk_filepath = output_dir_path / zarr_filename
    if spatial_chunk_filepath.exists():
        shutil.rmtree(spatial_chunk_filepath)

    time_intervals = pd.date_range(start=ds.time.values[0],
                                   end=ds.time.values[-1],
                                   freq=chunk_date_frequency)
    loading_writing_time = time()
    for i in range(len(time_intervals)):
        print(time_intervals[i])
        # Find range of indices of timesteps to cover
        bool_time_range_left = ds.time >= time_intervals[i].to_datetime64()
        if i < len(time_intervals) - 1:
            bool_time_range_right = ds.time <= time_intervals[i+1].to_datetime64()
        idxs = np.where(bool_time_range_left)[0] if i == len(time_intervals) - 1 else \
               np.where(bool_time_range_left & bool_time_range_right)[0]
        time_slice = slice(idxs[0], None) if i == len(time_intervals) - 1 else slice(idxs[0], idxs[-1])

        # Rechunk Zarr by pixel
        start_time = time()
        ds_sub = ds.isel(time=time_slice).compute()
        print("- Loading sliced dataset in memory: {:.0f}min".format((time() - start_time)/60))

        for var in ds_sub.data_vars:
            if "chunks" in ds_sub[var].encoding:
                del ds_sub[var].encoding['chunks']

        start_time = time()
        write_zarr(
            spatial_chunk_filepath.as_posix(),
            ds_sub,
            chunks=chunks,
            compressor=compressor,
            rounding=None,
            encoding=encoding,
            consolidated=True,
            append=(i != 0),
            append_dim="time",
            show_progress=True,
        )
        print("- Writing sliced spatially-chunked dataset: {:.0f}min\n".format((time() - start_time)/60))

        del ds_sub

    print("Rechunking Zarr per pixel: {:.0f}min".format((time() - loading_writing_time)/60))

    ds = xr.open_zarr(spatial_chunk_filepath)
    print("Total Zarr Dataset size when loaded: {:.0f}MB".format(ds.nbytes/10e6))


# ----------------------------------------------------------------------------.
if __name__ == "__main__":
    path_zarr = pathlib.Path("/ltenas3/data/NowProject/zarr/")
    ds = load_rzc_zarr(path_zarr / "rzc_temporal_chunk.zarr")
    rechunk_zarr_per_pixel(ds,
                           path_zarr,
                           chunks={"time": -1, "y": 5, "x": 5},
                           chunk_date_frequency="6MS",
                           compressor=zarr.Blosc(cname="zstd", clevel=3,
                                                 shuffle=2),
                           zarr_filename="rzc_spatial_chunk_5x5.zarr")
