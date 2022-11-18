import shutil
import pathlib
import datetime
from typing import Any, Union

from time import time
from warnings import warn
from zipfile import ZipFile
from trollsift import Parser
from itertools import repeat
from multiprocessing import Pool

import numpy as np
import xarray as xr
import pandas as pd

from pysteps.io.importers import import_mch_gif
from pyart.aux_io.metranet_cartesian_reader import read_cartesian_metranet
from pyart.aux_io import read_gif
from lte_toolbox.mch.archiving.utils import ensure_regular_timesteps
from lte_toolbox.mch.archiving.metadata import METADATA
from lte_toolbox.mch.archiving.encodings import (
    MCH_NETCDF_ENCODINGS,
    DATA_ENCODING_PER_PID,
    MCH_ZARR_ENCODINGS
)
from xforecasting.utils.zarr import write_zarr

DEFAULT_FILENAME_FORMAT = "{productid:3s}{time:%y%j%H%M}{radar_availability:2s}.{data_format}"
CPC_FILENAME_FORMAT = "{productid:3s}{time:%y%j%H%M}{radar_availability:s}_{accutime:s}.{data_format}.gif"

MCH_BOTTOM_LEFT_COORDINATES = [255, -160]
RADAR_BINARY_LOOKUP = {
    1: "A",
    2: "D",
    4: "L",
    8: "P",
    16: "W"
}

ASCII_CODE_POINT_A = ord("a")
MAX_NUMBER_IN_RADAR_LOOKUP_TABLE = 9

# -----------------------------------------------------------------------------.
# Unzipping


def unzip_mch(input_dir_path: pathlib.Path,
              output_dir_path: pathlib.Path,
              data_start_date: datetime.datetime,
              data_end_date: datetime.datetime = None,
              product: str = "RZC"):
    """Unzip all RZC .zip files for all years starting 2016 and save them
    in an output folder.

    Parameters
    ----------
    input_dir_path : pathlib.Path
        Path to input folder
    output_dir_path : pathlib.Path
        Path to folder where unzipped files will be saved
    data_start_year: int, optional
        Year starting which the data should be unzipped
    """
    if not data_end_date:
        data_end_date = datetime.datetime.today()
    folders = input_dir_path.glob("*")
    for folder_year in sorted(folders):
        year = int(folder_year.name)
        if year >= data_start_date.year:
            print(f"{year}.. ", end="")
            output_year_path = output_dir_path / str(year)
            output_year_path.mkdir(exist_ok=True)
            days = folder_year.glob("*")
            for folder_day in days:
                folder_datetime = datetime.datetime.strptime(folder_day.name, "%y%j")
                if folder_datetime >= data_start_date and folder_datetime <= data_end_date:
                    output_day_path = output_year_path / folder_day.name
                    output_day_path.mkdir(parents=True, exist_ok=True)
                    zip_path = list(folder_day.glob(f"{product}*.zip"))[0]
                    unzip_file(zip_path, output_day_path)
            print("done.")


def unzip_file(zip_path: pathlib.Path, output_path: pathlib.Path):
    """Unzip .zip file and save the foldeer in output_path.

    Parameters
    ----------
    input_path : pathlib.Path
        Path to input folder
    output_path : pathlib.Path
        Path to folder where unzipped files will be saved
    """
    with ZipFile(zip_path, 'r') as zip_ref:
        output_zip_path = output_path / zip_path.stem
        output_zip_path.mkdir(exist_ok=True)
        zip_ref.extractall(output_zip_path)


def unzip_files(input_path: pathlib.Path, output_path: pathlib.Path):
    """Unzip .zip files in input_path and save them in output_path.

    Parameters
    ----------
    input_path : pathlib.Path
        Path to input folder
    output_path : pathlib.Path
        Path to folder where unzipped files will be saved
    """
    for p in sorted(input_path.glob("*.zip")):
        with ZipFile(p, 'r') as zip_ref:
            zip_name = p.name[:-4]
            output_zip_path = output_path / zip_name
            output_zip_path.mkdir(exist_ok=True)
            zip_ref.extractall(output_zip_path)


# -----------------------------------------------------------------------------.
# Reading mch files

def read_mch_file(input_path: pathlib.Path, product_info: dict, **kwargs):
    if product_info.get("productid", "").lower() == "cpc":
        return import_mch_gif(input_path,
                              product_info.get("productid"),
                              "mm",
                              product_info.get("accutime"))
    else:
        return read_cartesian_metranet(input_path,
                                       reader=kwargs.get("reader", "python"))


def get_filename_format(filename: str) -> str:
    return CPC_FILENAME_FORMAT if "cpc" in filename.lower() else DEFAULT_FILENAME_FORMAT


def process_product_time(product_time: datetime.datetime) -> datetime.datetime:
    min = product_time.minute
    return product_time + datetime.timedelta(seconds=30) if\
        min == 2 or min == 7 else product_time


# TODO: Review radar quality and available radars
def get_numerical_radar_quality(radar_quality: str) -> Union[int, str]:
    if len(radar_quality) == 1:
        if radar_quality.isdigit():
            return int(radar_quality)
        else:
            return ord(radar_quality.lower()) - (ASCII_CODE_POINT_A - 1) \
                    + MAX_NUMBER_IN_RADAR_LOOKUP_TABLE
    else:
        return radar_quality


def get_available_radars(radar_quality: int) -> str:
    radars = ""
    for binary in RADAR_BINARY_LOOKUP:
        if binary & radar_quality:
            radars += RADAR_BINARY_LOOKUP[binary]

    return radars


def get_coords_from_corners(rzc):
    rzc_shape = rzc.shape
    x = np.arange(MCH_BOTTOM_LEFT_COORDINATES[0],
                  MCH_BOTTOM_LEFT_COORDINATES[0] + rzc_shape[1]) + 0.5
    y = np.arange(MCH_BOTTOM_LEFT_COORDINATES[1] + rzc_shape[0] - 1,
                  MCH_BOTTOM_LEFT_COORDINATES[1] - 1, -1) + 0.5
    x = x*1000
    y = y*1000

    return x, y


def get_metranet_header_dictionary(radar_filepath: str) -> dict:
    """Extracts the header of the RZC file.

    Parameters
    ----------
    radar_filepath : str
        Path to the RZC file.

    Returns
    -------
    dict
        Header of the RZC file containing the different metadata
    """
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


def get_data_header_from_mch(mch, input_path: pathlib.Path, product_info: dict):
    if product_info.get("productid", "").lower() in ["cpc", "cpch"]:
        header = mch[2]
        header["radar"] = get_available_radars(int(product_info["radar_availability"]))
        header["quality"] = get_numerical_radar_quality(product_info["radar_availability"])
        mch_data = np.ma.masked_where(mch[0] != np.nan,
                                      mch[0])
    else:
        header = get_metranet_header_dictionary(input_path)
        field = list(mch.fields.keys())[0]
        # Drop time dimension and flip data along y axis
        mch_data = mch.fields[field]["data"][0, :, :][::-1, :]

    return mch_data, header


def merge_metadata_product_header(product_header: dict):
    attrs = METADATA.copy()
    attrs["product"] = product_header.get("product", "")
    attrs["pid"] = product_header.get("pid", attrs["product"])
    attrs["accutime"] = float(product_header.get("accutime", 2.5))
    attrs["unit"] = product_header.get("unit", "") \
        if attrs["pid"].lower() in ["cpc", "cpch"] else \
        product_header.get("data_unit", "")

    return attrs


# TODO: Radar quality + attributes
def mch_file_to_xarray(input_path: pathlib.Path, **kwargs):
    p = Parser(get_filename_format(input_path.name))
    product_info = p.parse(input_path.name)
    product_info["time"] = process_product_time(product_info["time"])
    mch = read_mch_file(input_path, product_info, **kwargs)
    mch_data, header = get_data_header_from_mch(mch,
                                                input_path,
                                                product_info)
    # Compute CH1903 coordinates (at pixel centroids)
    # - Provide in meters for compatibility with Cartopy and unit of
    # CRS definition
    x, y = get_coords_from_corners(mch_data)

    # Create xr.Dataset
    ds = xr.Dataset(
                data_vars=dict(
                    data=(["y", "x"], mch_data.data),
                    mask=(["y", "x"], mch_data.mask.astype(int)),
                ),
                coords=dict(
                    time=process_product_time(product_info["time"]),
                    x=(["x"], x),
                    y=(["y"], y),
                    radar_availability=product_info["radar_availability"],
                    radar_names=header.get("radar", ""),
                    radar_quality=header.get("quality", "")
                ),
                attrs=merge_metadata_product_header(header),
            )
    return ds


def daily_mch_to_netcdf(input_dir_path: pathlib.Path,
                        output_dir_path: pathlib.Path,
                        file_suffix: str,
                        fill_value={"data": np.nan, "mask": 0},
                        **kwargs):
    """Read all the MCH files of a certain day, combine them into one xarray
    Dataset and save the result in a netCDF file.

    Parameters
    ----------
    input_dir_path : pathlib.Path
        Path to the input folder
    output_dir_path : pathlib.Path
        Path to the output folder
    file_suffix : str
        File suffix
    log_dir_path : pathlib.Path
        Path to save the logs of the reading process
    encoding : dict, optional
        Encoding of the netCDF file, by default NETCDF_ENCODINGS
    """
    output_filename = input_dir_path.name + ".nc"
    list_ds = []
    if not (output_dir_path / output_filename).exists():
        for file in sorted(input_dir_path.glob(f"*{file_suffix}")):
            try:
                list_ds.append(mch_file_to_xarray(file, **kwargs))
            except (TypeError, ValueError):
                continue
        if len(list_ds) > 0:
            time_offset = int(list_ds[0].attrs["accutime"] * 60)
            pid = list_ds[0].attrs["pid"].upper()
            encoding = MCH_NETCDF_ENCODINGS.copy()
            encoding["data"].update(DATA_ENCODING_PER_PID[pid])
            concat_ds = xr.concat(list_ds, dim="time", coords="all")
            concat_ds = ensure_regular_timesteps(concat_ds,
                                                 fill_value=fill_value,
                                                 t_res=f"{time_offset}s")
            concat_ds.to_netcdf(output_dir_path / output_filename,
                                encoding=encoding)


def produce_daily_netcdf(input_path: pathlib.Path,
                         output_zip_path: pathlib.Path,
                         output_netcdf_path: pathlib.Path,
                         file_suffix: str,
                         fill_value={"data": np.nan, "mask": 0}):
    with ZipFile(input_path, 'r') as zip_ref:
        zip_name = input_path.stem
        output_zip_day_path = output_zip_path / zip_name
        output_zip_day_path.mkdir(exist_ok=True)
        zip_ref.extractall(output_zip_day_path)
        daily_mch_to_netcdf(output_zip_day_path,
                            output_netcdf_path,
                            file_suffix,
                            fill_value)
        shutil.rmtree(output_zip_day_path)


def unzip_and_combine_mch(input_dir_path: pathlib.Path,
                          output_zip_path: pathlib.Path,
                          output_netcdf_path: pathlib.Path,
                          product: str,
                          file_suffix: str,
                          data_start_date: datetime.datetime,
                          data_end_date: datetime.datetime = None,
                          fill_value={"data": np.nan, "mask": 0},
                          num_workers: int = 2):
    if not data_end_date:
        data_end_date = datetime.datetime.today()
    folders = input_dir_path.glob("*")
    for folder_year in sorted(folders):
        year = int(folder_year.name)
        if year >= data_start_date.year:
            print(f"{year}.. ", end="")
            output_zip_year_path = output_zip_path / str(year)
            output_netcdf_year_path = output_netcdf_path / str(year)
            output_zip_year_path.mkdir(exist_ok=True)
            output_netcdf_year_path.mkdir(exist_ok=True)

            days = folder_year.glob("*")
            list_folders_days = []
            for folder_day in sorted(days):
                folder_datetime = datetime.datetime.strptime(folder_day.name, "%y%j")
                if folder_datetime >= data_start_date and folder_datetime <= data_end_date:
                    list_folders_days.append(list(folder_day.glob(f"{product}*.zip"))[0])

            with Pool(num_workers) as p:
                p.starmap(produce_daily_netcdf,
                          zip(list_folders_days,
                              repeat(output_zip_year_path),
                              repeat(output_netcdf_year_path),
                              repeat(file_suffix),
                              repeat(fill_value)))
            print("done.")

# -----------------------------------------------------------------------------.
# Zarr processing


def netcdf_mch_to_zarr(netcdf_dir_path: pathlib.Path,
                       output_dir_path: pathlib.Path,
                       compressor: Any = "auto",
                       encoding: dict = MCH_ZARR_ENCODINGS):
    temporal_chunk_filepath = output_dir_path / "chunked_by_time.zarr"
    if temporal_chunk_filepath.exists():
        shutil.rmtree(temporal_chunk_filepath)
    # --------------------------------------------------.
    # Load the dataset
    fpaths = [p.as_posix() for p in
              sorted(list(netcdf_dir_path.glob("*/*.nc")))]
    list_ds = [xr.open_dataset(p, chunks={"time": 576, "x": -1, "y": -1})
               for p in fpaths]
    ds = xr.concat(list_ds, dim="time")
    pid = list_ds[0].attrs["pid"].upper()
    encoding = MCH_NETCDF_ENCODINGS.copy()
    encoding["data"].update(DATA_ENCODING_PER_PID[pid])

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


def load_mch_zarr(rzc_zarr_filepath: pathlib.Path,
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
                           encoding: dict = MCH_ZARR_ENCODINGS,
                           zarr_filename: str = "mch_chunked_by_pixel.zarr"):
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


if __name__ == "__main__":
    zip_dir_path = pathlib.Path("/ltenas3/alfonso/msrad_bgg/2022")
    root_dir_path = pathlib.Path("/ltenas3/data/NowProject/snippet_mch/AZC/")
    if root_dir_path.exists():
        shutil.rmtree(root_dir_path)
    unzipped_dir_path = root_dir_path / "unzipped_temp"
    netcdf_dir_path = root_dir_path / "netcdf_temp"

    unzipped_dir_path.mkdir(exist_ok=True, parents=True)
    netcdf_dir_path.mkdir(exist_ok=True, parents=True)

    data_start_date = datetime.datetime.strptime("2022-01-01", "%Y-%m-%d")
    data_end_date = datetime.datetime.strptime("2022-06-15", "%Y-%m-%d")
    workers = 1
    unzip_and_combine_mch(zip_dir_path, unzipped_dir_path, netcdf_dir_path,
                          product="AZC", file_suffix=".801",
                          data_start_date=data_start_date,
                          num_workers=workers)

    # input_path = pathlib.Path("/ltenas3/data/NowProject/snippet_mch/2022/22152/AZC22152")
    # netcdf_dir_path = pathlib.Path("/ltenas3/data/NowProject/snippet_mch/AZC/netcdf_temp/2022")
    # ds = daily_mch_to_netcdf(input_path, netcdf_dir_path, file_suffix=".801")


