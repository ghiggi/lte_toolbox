#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 16:54:56 2022

@author: ghiggi
"""
import pathlib
from lte_mch_toolbox.archiving.rzc import (
    unzip_and_combine_rzc,
    unzip_rzc,
    rzc_to_netcdf,
    postprocess_all_netcdf,
    netcdf_rzc_to_zarr,
)

if __name__ == "__main__":
    # Likely the old one 
    zip_dir_path = pathlib.Path("/ltenas3/0_MCH/RZC/zipped")
    unzipped_dir_path = pathlib.Path("/ltenas3/0_MCH/RZC/unzipped_temp")
    netcdf_dir_path = pathlib.Path("/ltenas3/0_MCH/RZC/netcdf_temp/")
    log_dir_path = pathlib.Path("/ltenas3/0_MCH/RZC/logs_temp/")

    unzipped_dir_path.mkdir(exist_ok=True)
    netcdf_dir_path.mkdir(exist_ok=True)
    log_dir_path.mkdir(exist_ok=True)

    workers = 15
    unzip_and_combine_rzc(zip_dir_path, 
                          unzipped_dir_path, 
                          netcdf_dir_path, 
                          log_dir_path, 
                          num_workers=workers)
    
    # Likely the new one 
    unzipped_dir_path = pathlib.Path("/ltenas3/0_MCH/RZC/unzipped")
    netcdf_dir_path = pathlib.Path("/ltenas3/monika/data_lte/rzc/")  # TO BE MOVED IN 0_MCH RZC 
    postprocessed_netcdf_dir_path = pathlib.Path("/ltenas3/0_MCH/RZC/netcdf/")
    zarr_dir_path = pathlib.Path("/ltenas3/0_MCH/RZC/zarr/")
    log_dir_path = pathlib.Path("/ltenas3/0_MCH/RZC/logs/")

    workers = cpu_count() - 4
    unzip_rzc(zip_dir_path, unzipped_dir_path)
    rzc_to_netcdf(unzipped_dir_path, netcdf_dir_path, log_dir_path, num_workers=workers) 
    postprocess_all_netcdf(netcdf_dir_path, postprocessed_netcdf_dir_path, num_workers=workers)
    netcdf_rzc_to_zarr(postprocessed_netcdf_dir_path, 
                       zarr_dir_path,
                       compressor=zarr.Blosc(cname="zstd", clevel=3, shuffle=2))