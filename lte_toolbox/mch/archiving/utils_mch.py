import re
import shutil
import pathlib
import datetime
import tempfile
from typing import Any, Tuple, Union

from time import time
from warnings import warn
from zipfile import ZipFile
from trollsift import Parser
from itertools import repeat
from multiprocessing import Pool

import numpy as np
import xarray as xr
import pandas as pd

# from pysteps.io.importers import import_mch_gif
from pyart.aux_io.metranet_cartesian_reader import read_cartesian_metranet
from pyart.aux_io import read_gif
from pyart.core import Grid
from lte_toolbox.mch.archiving.utils_timesteps import ensure_regular_timesteps
from lte_toolbox.mch.archiving.metadata import METADATA
from lte_toolbox.mch.archiving.encodings import (
    MCH_NETCDF_ENCODINGS,
    DATA_ENCODING_PER_PID,
    MCH_ZARR_ENCODINGS
)
from xforecasting.utils.zarr import write_zarr

# -----------------------------------------------------------------------------.
# Constant variables


DEFAULT_FILENAME_FORMAT = "{productid:3s}{time:%y%j%H%M}{radar_availability:2s}.{data_format}"
CPC_FILENAME_FORMAT = "{productid:3s}{time:%y%j%H%M}{radar_availability:s}_{accutime:s}.{data_format}.gif"

AVAILABLE_VALID_SUFFIXES = {
    "AZC": [".801"],
    "BZC": [".845"],
    "CPC": [".801.gif"],
    "CPCH": [".801.gif"],
    "CZC": [".801"],
    "EZC": [".815", ".820", ".845", ".850"],
    "LZC": [".801"],
    "MZC": [".850"],
    "RZC": [".801"],
    "aZC": [".824"]
}

AVAILABLE_VALID_ACCUTIME = [5, 60, 180, 360, 720, 1440, 2880, 4320]

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

SHORTENED_UNIT = {"millimeter": "mm", "millimeter/hour": "mm/h"}

# -----------------------------------------------------------------------------.
# Reading mch files


def read_mch_file(input_path: pathlib.Path, product_info: dict, **kwargs) -> Grid:
    """Read the content of a MCH file using the metranet or the gif reader,
    depending on the product type.

    Parameters
    ----------
    input_path : pathlib.Path
        Path to MCH file
    product_info : dict
        Dictionary containing product info, should contain
        at least "productid" and "accutime"

    Returns
    -------
    Grid
        Grid object containing the data extracted by the reader
    """
    if product_info.get("productid", "").lower() == "cpc":
        # return import_mch_gif(input_path,
        #                       product_info.get("productid"),
        #                       "mm",
        #                       product_info.get("accutime"))
        return read_gif(input_path)
    else:
        return read_cartesian_metranet(input_path,
                                       reader=kwargs.get("reader", "python"))


def get_filename_format(filename: str) -> str:
    """Return the format of the filename to extract relevant information
    from it.

    Parameters
    ----------
    filename : str
        Filename to obtain the format of

    Returns
    -------
    str
        Format of the filename
    """
    return CPC_FILENAME_FORMAT if "cpc" in filename.lower() else DEFAULT_FILENAME_FORMAT


def process_product_time(product_time: datetime.datetime) -> datetime.datetime:
    """Add half a minute to the time associated to a measurement if the
    last digit is either 2 or 7. The presence of a 2 or 7 generally translates
    in a time resolution of 2.5 mins.

    Parameters
    ----------
    product_time : datetime.datetime
        Datetime to process

    Returns
    -------
    datetime.datetime
        Processed datetime
    """
    last_digit_min = product_time.minute % 10
    return product_time + datetime.timedelta(seconds=30) if\
        last_digit_min == 2 or last_digit_min == 7 else product_time


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


def get_coords_from_corners(mch_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return the list of x and y coordinates for each point in the grid
    from the predefined MCH corner coordinates.

    Parameters
    ----------
    mch_data : np.ndarray
        MCH data to get the coordinates for

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Lists of x and y coordinates corresponding to each grid point
    """
    mch_shape = mch_data.shape
    x = np.arange(MCH_BOTTOM_LEFT_COORDINATES[0],
                  MCH_BOTTOM_LEFT_COORDINATES[0] + mch_shape[1]) + 0.5
    y = np.arange(MCH_BOTTOM_LEFT_COORDINATES[1] + mch_shape[0] - 1,
                  MCH_BOTTOM_LEFT_COORDINATES[1] - 1, -1) + 0.5
    x = x*1000
    y = y*1000

    return x, y


def get_metranet_header_dictionary(radar_filepath: str) -> dict:
    """Extracts the header of the MCH file. The product file must not
    be a gif.

    Parameters
    ----------
    radar_filepath : str
        Path to the RZC file.

    Returns
    -------
    dict
        Header of the MCH file containing the different metadata
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


def get_data_header_from_gif(mch_gif: Grid) -> dict:
    """Given a Grid object returned by Pyart_mch's gif reader, extract
    the relevant metadata to include in the data header (product, product id,
    accutime and unit).

    Parameters
    ----------
    mch_gif : Grid
        Grid object returned by Pyart_mch gif reader

    Returns
    -------
    dict
        Dictionary containing the metadata header linked to the data
    """
    header = {}
    header["product"] = mch_gif.metadata["long_name"]
    header["pid"] = re.findall(r'([a-zA-Z ]*)\d*.*', mch_gif.metadata["PID"])[0]
    header["accutime"] = mch_gif.metadata["ACCUm"]
    header["unit"] = SHORTENED_UNIT.get(mch_gif.metadata["units"],
                                        mch_gif.metadata["units"])

    return header


def get_data_header_from_mch(mch, input_path: pathlib.Path,
                             product_info: dict) -> Tuple[np.ma.MaskedArray, dict]:
    """Return the data and the header associated to an MCH file.

    Parameters
    ----------
    mch : _type_
        Object extracted by the reader
    input_path : pathlib.Path
        Path to MCH file
    product_info : dict
        Dictionary containing product info, should contain
        at least "productid" and "radar_availability"

    Returns
    -------
    Tuple[np.ma.MaskedArrray, dict]
        Data and header associated to an MCH file.
    """
    field = list(mch.fields.keys())[0]
    mch_data = mch.fields[field]["data"][0, :, :][::-1, :]
    if product_info.get("productid", "").lower() in ["cpc", "cpch"]:
        mch_data = mch_data.copy()
        mch_data[mch_data == 255] = np.nan
        mch_data = np.ma.masked_where(mch_data.data != np.nan,
                                      mch_data.data)

        header = get_data_header_from_gif(mch)
        header["radar"] = get_available_radars(int(product_info["radar_availability"]))
        header["quality"] = get_numerical_radar_quality(product_info["radar_availability"])
    else:
        header = get_metranet_header_dictionary(input_path)

    return mch_data, header


def merge_metadata_product_header(product_header: dict) -> dict:
    """Merge basic metadata for an MCH product and the header
    extracted from the product file.

    Parameters
    ----------
    product_header : dict
        Header associated to the MCH file

    Returns
    -------
    dict
        Merged metadata dictionary
    """
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
                        accutime: int = None,
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
    fill_value : dict, optional
        Dictionary containing the fill values for each variable in
        case data is missing, by default {"data": np.nan, "mask": 0}
    """
    if accutime:
        accutime = "{:05d}".format(accutime)

    output_filename = input_dir_path.name + ".nc"
    list_ds = []
    if not (output_dir_path / output_filename).exists():
        filename_pattern = f"*{accutime}{file_suffix}" if accutime \
                            else f"*{file_suffix}"
        for file in sorted(input_dir_path.glob(filename_pattern)):
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


def pipeline_daily_mch(input_zip_path: pathlib.Path,
                       output_zip_path: pathlib.Path,
                       output_netcdf_path: pathlib.Path,
                       file_suffix: str,
                       accutime: int = None,
                       fill_value={"data": np.nan, "mask": 0}):
    """Unzip MCH daily zip file, then read and combine all the
    extracted files into one netCDF file.

    Parameters
    ----------
    input_zip_path : pathlib.Path
        Path to the input zip file
    output_zip_path : pathlib.Path
        Path to folder where the files will be unzipped
    output_netcdf_path : pathlib.Path
        Path to folder where the netCDF will be saved
    file_suffix : str
        File suffix
    fill_value : dict, optional
        Dictionary containing the fill values for each variable in
        case data is missing, by default {"data": np.nan, "mask": 0}
    """
    with ZipFile(input_zip_path, 'r') as zip_ref:
        zip_name = input_zip_path.stem
        output_zip_day_path = output_zip_path / zip_name
        output_zip_day_path.mkdir(exist_ok=True)
        zip_ref.extractall(output_zip_day_path)
        daily_mch_to_netcdf(output_zip_day_path,
                            output_netcdf_path,
                            file_suffix,
                            accutime=accutime,
                            fill_value=fill_value)
        shutil.rmtree(output_zip_day_path)


def unzip_and_combine_mch(base_dir: pathlib.Path,
                          temp_dir_netcdf: pathlib.Path,
                          product: str,
                          file_suffix: str,
                          data_start_date: datetime.datetime,
                          data_end_date: datetime.datetime = None,
                          accutime: int = None,
                          fill_value={"data": np.nan, "mask": 0},
                          num_workers: int = 2):
    """In parallel, unzip all MCH daily zip files contained in the hierarchy, then
    read and combine all the extracted files into daily netCDF files.

    Parameters
    ----------
    base_dir : pathlib.Path
        Path to the input folder
    temp_dir_netcdf : pathlib.Path
        Path to folder where the netCDF files will be saved
    product : str
        Product ID
    file_suffix : str
        File suffix
    data_start_date : datetime.datetime
        Start date of the desired time period to extract
    data_end_date : datetime.datetime, optional
        End date of the desired time period to extract, by default None
    accutime : int
        Accumulation time
    fill_value : dict, optional
        Dictionary containing the fill values for each variable in
        case data is missing, by default {"data": np.nan, "mask": 0}
    num_workers : int, optional
        Number of workers to parallelize the extraction and combination
        process, by default 2
    """
    if product not in AVAILABLE_VALID_SUFFIXES:
        raise ValueError("The product you tried to read is not taken into account yet.")

    if file_suffix not in AVAILABLE_VALID_SUFFIXES[product]:
        raise ValueError("The file suffix you provided is not in the list of valid suffixes for this product.")

    if accutime and accutime not in AVAILABLE_VALID_ACCUTIME:
        raise ValueError("The accumulation time indicated is not correct.")

    if accutime and product not in ["CPC", "CPCH"]:
        accutime = None
        print("Accutime set to None as product is neither CPC nor CPCH.")

    if not data_end_date:
        data_end_date = datetime.datetime.today()

    output_zip_path = pathlib.Path(tempfile.mkdtemp())
    folders = base_dir.glob("*")
    for folder_year in sorted(folders):
        year = int(folder_year.name)
        if year >= data_start_date.year:
            print(f"{year}.. ", end="")
            output_zip_year_path = output_zip_path / str(year)
            output_netcdf_year_path = temp_dir_netcdf / str(year)
            output_zip_year_path.mkdir(exist_ok=True)
            output_netcdf_year_path.mkdir(exist_ok=True)

            days = folder_year.glob("*")
            list_folders_days = []
            for folder_day in sorted(days):
                folder_datetime = datetime.datetime.strptime(folder_day.name, "%y%j")
                if folder_datetime >= data_start_date and folder_datetime <= data_end_date:
                    list_folders_days.append(list(folder_day.glob(f"{product}*.zip"))[0])

            with Pool(num_workers) as p:
                p.starmap(pipeline_daily_mch,
                          zip(list_folders_days,
                              repeat(output_zip_year_path),
                              repeat(output_netcdf_year_path),
                              repeat(file_suffix),
                              repeat(accutime),
                              repeat(fill_value)))
            print("done.")

    shutil.rmtree(output_zip_path)


# -----------------------------------------------------------------------------.
# Zarr processing


def netcdf_mch_to_zarr(netcdf_dir_path: pathlib.Path,
                       output_dir_path: pathlib.Path,
                       chunks: dict = {"time": 576, "x": -1, "y": -1},
                       compressor: Any = "auto"):
    """Load all the daily netCDF files in a hierarchy and combine them into
    one Zarr file, chunked by time.

    Parameters
    ----------
    netcdf_dir_path : pathlib.Path
        Path to parent folder containing the netCDF files.
        Structure parent_folder/year/*.nc.
    output_dir_path : pathlib.Path
        Path to folder where the Zarr file will be saved.
    chunks: dict

    compressor : Any, optional
        Compressor to use when saving the Zarr file, by default "auto"
    """
    temporal_chunk_filepath = output_dir_path / "chunked_by_time.zarr"
    if temporal_chunk_filepath.exists():
        shutil.rmtree(temporal_chunk_filepath)
    # --------------------------------------------------.
    # Load the dataset
    fpaths = [p.as_posix() for p in
              sorted(list(netcdf_dir_path.glob("*/*.nc")))]
    list_ds = [xr.open_dataset(p, chunks=chunks)
               for p in fpaths]
    ds = xr.concat(list_ds, dim="time")
    pid = list_ds[0].attrs["pid"].upper()
    encoding = MCH_ZARR_ENCODINGS.copy()
    encoding["data"] = DATA_ENCODING_PER_PID[pid]
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


def load_mch_zarr(mch_zarr_filepath: pathlib.Path,
                  mask_to_uint: bool = True) -> xr.Dataset:
    """Open the Zarr file containing the MCH data, and assign the
    correct types to coordinates and the mask variable.

    Parameters
    ----------
    mch_zarr_filepath : pathlib.Path
        Path to the Zarr file to open
    mask_to_uint : bool, optional
        Whether to convert the mask variable to uint8, by default True

    Returns
    -------
    xr.Dataset
        Processed xarray dataset with the correct coordinates and
        variables type
    """
    ds = xr.open_zarr(mch_zarr_filepath)
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
                           zarr_filename: str = "chunked_by_pixel.zarr"):
    """Rechunk pixel-wise a temporally chunked xarray dataset and save it in
    Zarr format.

    Parameters
    ----------
    ds : xr.Dataset
        Temporally chunked dataset to rechunk
    output_dir_path : pathlib.Path
        Path to folder where the Zarr file will be saved.
    chunks : dict
        Dictionary containing chunks per coordinate.
        E.g. {"time": -1, "x": 1, "y": 1}
    chunk_date_frequency : str, optional
        Temporal length of each spatial chunk, by default "6MS"
    compressor : Any, optional
        Compressor to use when saving the Zarr file, by default "auto"
    zarr_filename : str, optional
        Zarr filename, by default "chunked_by_pixel.zarr"
    """
    spatial_chunk_filepath = output_dir_path / zarr_filename
    if spatial_chunk_filepath.exists():
        shutil.rmtree(spatial_chunk_filepath)

    time_intervals = pd.date_range(start=ds.time.values[0],
                                   end=ds.time.values[-1],
                                   freq=chunk_date_frequency)
    loading_writing_time = time()
    pid = ds.attrs["pid"].upper()
    encoding = MCH_ZARR_ENCODINGS.copy()
    encoding["data"] = DATA_ENCODING_PER_PID[pid]

    for i in range(len(time_intervals)):
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
