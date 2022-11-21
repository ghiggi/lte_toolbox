import zarr
import shutil
import pathlib
import datetime

from lte_toolbox.mch.archiving.utils_mch import (
    unzip_and_combine_mch,
    netcdf_mch_to_zarr,
    load_mch_zarr,
    rechunk_zarr_per_pixel
)

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

product = "AZC"

zip_dir_path = pathlib.Path("/ltenas3/alfonso/msrad_bgg/2022")
root_dir_path = pathlib.Path(f"/ltenas3/data/NowProject/snippet_mch/{product}/")
if root_dir_path.exists():
    shutil.rmtree(root_dir_path)

unzipped_dir_path = root_dir_path / "unzipped_temp"
netcdf_dir_path = root_dir_path / "netcdf_temp"
zarr_dir_path = root_dir_path / "zarr"

unzipped_dir_path.mkdir(exist_ok=True, parents=True)
netcdf_dir_path.mkdir(exist_ok=True, parents=True)
zarr_dir_path.mkdir(exist_ok=True, parents=True)

data_start_date = datetime.datetime.strptime("2022-01-01", "%Y-%m-%d")
data_end_date = datetime.datetime.strptime("2022-12-31", "%Y-%m-%d")
workers = 12

print("MCH Extraction")
unzip_and_combine_mch(zip_dir_path, unzipped_dir_path, netcdf_dir_path,
                      product=product, file_suffix=file_suffixes[product],
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
