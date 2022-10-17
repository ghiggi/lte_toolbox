#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 15:14:44 2022

@author: ghiggi
"""
import shutil
import pathlib
import datetime

import pyart
import numpy as np
import xarray as xr 
from warnings import warn
from typing import Any
from zipfile import ZipFile
from itertools import repeat
from multiprocessing import Pool

from lte_toolbox.mch.archiving.encodings import RZC_NETCDF_ENCODINGS, RZC_ZARR_ENCODINGS
from lte_toolbox.mch.archiving.metadata import RZC_METADATA
from lte_toolbox.mch.archiving.utils import xr_drop_duplicated_timesteps 
from lte_toolbox.mch.archiving.utils import xr_regularize_time_dimension 


RZC_BOTTOM_LEFT_COORDINATES = [255, -160]

def get_metranet_header_dictionary(radar_filepath):
    # Example
    # fpath = "/home/ghiggi/RZC203221757VL.801"
    # radar_file = fpath 
    # header_dict = get_metranet_header_dictionary(fpath)
    prd_header = {'row': 0, 'column': 0}
    try:
       with open(radar_filepath, 'rb') as data_file:
           for t_line in data_file:
               line = t_line.decode("utf-8").strip('\n')
               if line.find('end_header') == -1:
                   data = line.split('=')
                   prd_header[data[0]] = data[1]
               else:
                   break
       return prd_header   
    except OSError as ee:
        warn(str(ee))
        print("Unable to read file '%s'" % radar_filepath)
        return None
    

def get_time_from_rzc_filename(filename: str):
    time = datetime.datetime.strptime(filename[3:12], "%y%j%H%M")
    if filename[3:12].endswith("2") or filename[3:12].endswith("7"):
        time = time + datetime.timedelta(seconds=30)
    return time


def read_rzc_file(input_path: pathlib.Path,  
                  row_start: int = 0, row_end: int = 640, 
                  col_start: int = 0, col_end: int = 710) -> xr.Dataset:
    # Get cartesian metranet class instance
    metranet = pyart.aux_io.read_cartesian_metranet(input_path.as_posix())
    # Get RZC array
    rzc = metranet.fields['radar_estimated_rain_rate']['data'][0,:,:]
    # Get RZC metadata
    metranet_header = get_metranet_header_dictionary(input_path.as_posix())
    # Compute CH1903 coordinates (at pixel centroids)
    # - Provide in meters for compatibility with Cartopy and unit of CRS definition
    rzc_shape = rzc.shape
    x = np.arange(RZC_BOTTOM_LEFT_COORDINATES[0], RZC_BOTTOM_LEFT_COORDINATES[0] + rzc_shape[1]) + 0.5
    y = np.arange(RZC_BOTTOM_LEFT_COORDINATES[1] + rzc_shape[0] - 1, RZC_BOTTOM_LEFT_COORDINATES[1] - 1, -1) + 0.5
    x = x*1000
    y = y*1000
    # Get timestep from filename 
    time = get_time_from_rzc_filename(input_path.as_posix().split("/")[-1])
    # Get radar availability
    radar_availability = input_path.as_posix().split("/")[-1][12:14]
    # Create xr.Dataset
    ds = xr.Dataset(
                data_vars=dict(
                    precip=(["y", "x"], rzc.data),
                    mask=(["y", "x"], rzc.mask.astype(int)),
                ),
                coords=dict(
                    time=time,
                    x=(["x"], x),
                    y=(["y"], y),
                    radar_availability=radar_availability,
                    radar_names=metranet_header.get("radar", ""),
                    radar_quality=metranet_header.get("quality", "")
                ),
                attrs={},
            )
    return ds

def daily_rzc_data_to_netcdf(input_dir_path: pathlib.Path,
                             output_dir_path: pathlib.Path,
                             log_dir_path: pathlib.Path,
                             encoding: dict = RZC_NETCDF_ENCODINGS, 
                             row_start: int = 0, 
                             row_end: int = 640, 
                             col_start: int = 0, 
                             col_end: int = 710):
    
    filename = input_dir_path.as_posix().split("/")[-1]
    output_filename = filename + ".nc"
    list_ds = []
    if not (output_dir_path / output_filename).exists():
        for file in sorted(input_dir_path.glob("*.801")):
            try:
                list_ds.append(read_rzc_file(file, row_start, row_end, col_start, col_end))
            except TypeError:
                with open(log_dir_path / f"type_errors_{filename}.txt", "a+") as f:
                    f.write(f"{file.as_posix()}\n")
            except ValueError:
                with open(log_dir_path / f"value_errors_{filename}.txt", "a+") as f:
                    f.write(f"{file.as_posix()}\n")

        if len(list_ds) > 0:
            xr.concat(list_ds, dim="time", coords="all")\
              .to_netcdf(output_dir_path / output_filename, encoding=encoding)


def rzc_to_netcdf(data_dir_path: pathlib.Path,
                  output_dir_path: pathlib.Path, 
                  log_dir_path: pathlib.Path, 
                  num_workers: int = 6):
    for folder_year in sorted(data_dir_path.glob("*")):
        year = int(folder_year.as_posix().split("/")[-1])
        print(year)
        output_dir_year_path = output_dir_path / str(year)
        output_dir_year_path.mkdir(exist_ok=True)

        list_folders_days = list(sorted(folder_year.glob("*")))
        with Pool(num_workers) as p:
            p.starmap(daily_rzc_data_to_netcdf, 
                      zip(list_folders_days, repeat(output_dir_year_path), repeat(log_dir_path)))


def postprocess_netcdf_file(input_file_path: pathlib.Path, 
                            output_dir_year_path: pathlib.Path, 
                            encoding: dict = RZC_NETCDF_ENCODINGS):
    from xarray.core import dtypes
    
    ds = xr.open_dataset(input_file_path).sortby("time")
    ds = xr_drop_duplicated_timesteps(ds)
   
    fill_value={"precip": dtypes.NA, "mask": 0}
    ds = xr_regularize_time_dimension(ds,
                                      t_res="2min30s", 
                                      fill_value=fill_value)
    ds['radar_quality'] = ds['radar_quality'].astype(str)
    ds['radar_availability'] = ds['radar_availability'].astype(str)
    ds['radar_names'] = ds['radar_names'].astype(str)

    output_filename = input_file_path.as_posix().split("/")[-1]
    ds.to_netcdf(output_dir_year_path / output_filename, encoding=encoding)


def postprocess_all_netcdf(data_dir_path: pathlib.Path, output_dir_path: pathlib.Path, num_workers: int = 6):
    for year_dir_path in data_dir_path.glob("*"):
        year = int(year_dir_path.as_posix().split("/")[-1])
        print(year)
        output_dir_year_path = output_dir_path / str(year)
        output_dir_year_path.mkdir(exist_ok=True)
        fpaths = list(year_dir_path.glob("*"))

        with Pool(num_workers) as p:
            p.starmap(postprocess_netcdf_file, zip(fpaths, repeat(output_dir_year_path)))
            

def netcdf_rzc_to_zarr(data_dir_path: pathlib.Path, 
                       output_dir_path: pathlib.Path, 
                       compressor: Any = "auto", 
                       encoding: dict = RZC_ZARR_ENCODINGS):
    from xforecasting.utils.zarr import rechunk_Dataset, write_zarr
    
    temporal_chunk_filepath = output_dir_path / "chunked_by_time.zarr"
    if temporal_chunk_filepath.exists():
        shutil.rmtree(temporal_chunk_filepath)
        
    ###--------------------------------------------------.
    # Load the dataset
    fpaths = [p.as_posix() for p in sorted(list(data_dir_path.glob("*/RZC*.nc")))]
    list_ds = [xr.open_dataset(p, chunks={"time": 576, "x": -1, "y": -1}) for p in fpaths]
    ds = xr.concat(list_ds, dim="time")
    ds.attrs = RZC_METADATA # TODO: MAYBE ADD ALREADY TO NETCDF
    
    ###--------------------------------------------------.
    ### Rechunk Zarr by block of time  
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
    
    
    ds = xr.open_zarr(temporal_chunk_filepath)
    ds['radar_quality'] = ds['radar_quality'].astype(str)
    ds['radar_availability'] = ds['radar_availability'].astype(str)
    ds['radar_names'] = ds['radar_names'].astype(str)
    ###--------------------------------------------------.
    ### Rechunk Zarr by pixel 
    spatial_chunk_filepath = output_dir_path / "chunked_by_pixel.zarr"
    spatial_chunk_temp_filepath = output_dir_path / "chunked_by_pixel_temp.zarr"
    if spatial_chunk_filepath.exists():
        shutil.rmtree(spatial_chunk_filepath)
    if spatial_chunk_temp_filepath.exists():
        shutil.rmtree(spatial_chunk_temp_filepath)

    rechunk_Dataset(ds, {"time": -1, "y": 1, "x": 1},
                    spatial_chunk_filepath.as_posix(), 
                    spatial_chunk_temp_filepath.as_posix(), 
                    max_mem="1GB", force=False)
    return None
    
####-----------------------------------------------------------------------------.
#### Unzipping  
def unzip_rzc(input_dir_path: pathlib.Path, 
              output_dir_path: pathlib.Path):
    folders = input_dir_path.glob("*")
    for folder in sorted(folders):
        year = int(folder.as_posix().split("/")[-1])
        if year >= 2016:
            print(f"{year}.. ", end="")
            output_year_path = output_dir_path / str(year)
            output_year_path.mkdir(exist_ok=True)
            unzip_files(folder, output_year_path)
            print("done.") 


def unzip_files(input_path: pathlib.Path, output_path: pathlib.Path):
    for p in sorted(input_path.glob("*.zip")):
        with ZipFile(p, 'r') as zip_ref:
            zip_name = p.as_posix().split("/")[-1][:-4]
            output_zip_path = output_path / zip_name
            output_zip_path.mkdir(exist_ok=True)
            zip_ref.extractall(output_zip_path)

####-----------------------------------------------------------------------------.
### TODO: OLD? 
def unzip_and_combine_day(input_path: pathlib.Path, 
                          output_zip_path: pathlib.Path, 
                          output_netcdf_path: pathlib.Path, 
                          log_dir_path: pathlib.Path):
    with ZipFile(input_path, 'r') as zip_ref:
        zip_name = input_path.as_posix().split("/")[-1][:-4]
        output_zip_day_path = output_zip_path / zip_name
        output_zip_day_path.mkdir(exist_ok=True)

        zip_ref.extractall(output_zip_day_path)
        daily_rzc_data_to_netcdf(output_zip_day_path, output_netcdf_path, log_dir_path)
        shutil.rmtree(output_zip_day_path.as_posix())


def unzip_and_combine_rzc(input_dir_path: pathlib.Path, 
                          output_zip_path: pathlib.Path,
                          output_netcdf_path: pathlib.Path,
                          log_dir_path: pathlib.Path,
                          num_workers: int = 2):
    folders = input_dir_path.glob("*")
    for folder in sorted(folders):
        year = int(folder.as_posix().split("/")[-1])
        if year >= 2016:
            print(f"{year}.. ")
            output_zip_year_path = output_zip_path / str(year)
            output_netcdf_year_path = output_netcdf_path / str(year)
            output_zip_year_path.mkdir(exist_ok=True)
            output_netcdf_year_path.mkdir(exist_ok=True)

            list_folders_days = list(sorted(folder.glob("*.zip")))
            with Pool(num_workers) as p:
                p.starmap(unzip_and_combine_day, 
                          zip(list_folders_days, repeat(output_zip_year_path), 
                              repeat(output_netcdf_year_path), repeat(log_dir_path)))


####-----------------------------------------------------------------------------.
