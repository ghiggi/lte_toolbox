import zarr
import time
import shutil
import pathlib
import datetime

from lte_toolbox.mch.archiving.utils_mch import (
    unzip_and_combine_mch,
    netcdf_mch_to_zarr,
    load_mch_zarr,
    rechunk_zarr_per_pixel
)

# TODO: check suffixes from past
# TODO: See how to generalise code to other coordinate systems
# TODO: add function to update zarr when new data is available

file_suffixes = {
    "AZC": ".801",
    "BZC": ".845",
    "CPC": ".801.gif",
    "CPCH": ".801.gif",
    "CZC": ".801",
    "EZC": ".815",
    "LZC": ".801",
    "MZC": ".850",
    "RZC": ".801",
    "aZC": ".824"
}

accutimes = {
    "AZC": None,
    "BZC": None,
    "CPC": 60,
    "CPCH": 60,
    "CZC": None,
    "EZC": None,
    "LZC": None,
    "MZC": None,
    "RZC": None,
    "aZC": None
}

for product in file_suffixes:
    t_start = time.time()
    print(product)
    # Point at directory with all the yearly subfolders (2020, 2021, ..)
    zip_dir_path = pathlib.Path("/ltenas8/mch/msrad/")
    # Point at directory where you want to save your data
    root_dir_path = pathlib.Path(f"/ltenas8/data/NowProject/mch_zarr/{product}")
    if accutimes[product] is not None:
        root_dir_path = root_dir_path / str(accutimes[product])

    if root_dir_path.exists():
        shutil.rmtree(root_dir_path)

    root_dir_path.mkdir(parents=True, exist_ok=True)

    netcdf_dir_path = root_dir_path / "netcdf_temp"
    zarr_dir_path = root_dir_path / "zarr"

    netcdf_dir_path.mkdir(exist_ok=True, parents=True)
    zarr_dir_path.mkdir(exist_ok=True, parents=True)

    data_start_date = datetime.datetime.strptime("2022-01-01", "%Y-%m-%d")
    data_end_date = datetime.datetime.strptime("2022-12-31", "%Y-%m-%d")
    workers = 12

    print("MCH Extraction")
    unzip_and_combine_mch(zip_dir_path, netcdf_dir_path,
                          product=product, file_suffix=file_suffixes[product],
                          accutime=accutimes[product],
                          data_start_date=data_start_date,
                          num_workers=workers)

    print("Converting to temporally-chunked zarr")
    netcdf_mch_to_zarr(netcdf_dir_path, zarr_dir_path,
                       compressor=zarr.Blosc(cname="zstd", clevel=3, shuffle=2))

    print("Temporal to spatial chunking")
    ds = load_mch_zarr(zarr_dir_path / "chunked_by_time.zarr")
    rechunk_zarr_per_pixel(ds,
                           zarr_dir_path,
                           chunks={"time": -1, "y": 5, "x": 5},
                           chunk_date_frequency="6MS",
                           compressor=zarr.Blosc(cname="zstd", clevel=3,
                                                 shuffle=2),
                           zarr_filename="chunked_by_pixel_5x5.zarr")
    print("\nTotal time to process {}: {:.0f}min".format(product, (time.time() - t_start)/60))
    print()
