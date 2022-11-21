import re
import pathlib
import pyart
import numpy as np
import xarray as xr
from lte_toolbox.mch.archiving import utils_mch
from lte_toolbox.mch.archiving.utils import ensure_regular_timesteps
from lte_toolbox.mch.archiving.encodings import (
    MCH_NETCDF_ENCODINGS,
    DATA_ENCODING_PER_PID
)
from trollsift import Parser

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

path_input_mch = pathlib.Path("/ltenas3/alfonso/msrad_bgg/2022/")
path_output_mch = pathlib.Path("/ltenas3/data/NowProject/snippet_mch/")

utils_mch.unzip_metranet(path_input_mch, path_output_mch)

# AZC2215200007L.801
# BZC221520000VL.845
# CPC2215200003_02880.801.gif
# CPC2215200003_02880.801.gif
# CZC221520000VL.801
# EZC221520000VL.815
# LZC221520000VL.801
# MZC221520000VL.850
# RZC221520000VL.801
# aZC2215200007L.824

products = {}
path_input_day = path_output_mch / "2022" / "22152"
for product in path_input_day.glob("*"):
    product_example_input = list(product.glob("*"))[0]
    product_name = re.findall(r'([a-zA-Z ]*)\d*.*', product.name)[0]
    products[product_name] = {"filename": product_example_input.name}
    if product_name in ["CPC", "CPCH", "aZC"]:
        continue
    print(product_name)
    metranet = pyart.aux_io.read_cartesian_metranet(
        product_example_input.as_posix(), reader="python")
    products[product_name]["field"] = list(metranet.fields.keys())[0]
    products[product_name]["data_shape"] = \
        metranet.fields[products[product_name]["field"]]["data"][0, :, :].shape

products = {}
for product in path_input_day.glob("*"):
    product_name = re.findall(r'([a-zA-Z ]*)\d*.*', product.name)[0]
    print(product_name)
    product_example_input = list(product.glob("*"))[0]
    p = Parser(utils_mch.get_filename_format(product_name))
    products[product_name] = p.parse(product_example_input.name)


filepaths = []
for product in path_input_day.glob("*"):
    filepaths.append(list(product.glob("*"))[0])

ds_products = []
ds_headers = {}
kwargs = {"reader": "python"}
for input_path in filepaths[:-1]:
    print(input_path.name)
    ds_products.append(utils_mch.mch_file_to_xarray(input_path, reader="python"))

for i, ds in enumerate(ds_products):
    print(filepaths[i].name)
    print(f"{ds.attrs['product']}, {ds.attrs['pid']}, {ds.attrs['unit']}")
    unique = np.unique(ds.data.values)
    print(unique)


ds_cpc = []
for product in path_input_day.glob("*"):
    if "CPC" in product.name:
        for path in product.glob("*"):
            ds_cpc.append(utils_mch.mch_file_to_xarray(path, reader="python"))


ds_products = []
root_path = pathlib.Path("/ltenas3/data/NowProject/snippet_mch/2022/22152/")
for product in root_path.glob("*"):
    product_name = re.findall(r'([a-zA-Z ]*)\d*.*', product.name)[0]
    print(product_name)
    list_ds = []
    for file in sorted(product.glob(f"*{file_suffixes[product_name]}")):
        try:
            list_ds.append(utils_mch.mch_file_to_xarray(file))
        except (TypeError, ValueError):
            continue
    if len(list_ds) > 0:
        time_offset = int(list_ds[0].attrs["accutime"] * 60)
        pid = list_ds[0].attrs["pid"].upper()
        encoding = MCH_NETCDF_ENCODINGS.copy()
        encoding["data"].update(DATA_ENCODING_PER_PID[pid])
        concat_ds = xr.concat(list_ds, dim="time", coords="all")
        concat_ds = ensure_regular_timesteps(concat_ds,
                                             fill_value={"data": np.nan, "mask": 0},
                                             t_res=f"{time_offset}s")

    ds_products.append(concat_ds)


for i, product in enumerate(root_path.glob("*")):
    product_name = re.findall(r'([a-zA-Z ]*)\d*.*', product.name)[0]
    print(product_name)
    print(f"{ds_products[i].attrs['product']}, {ds_products[i].attrs['pid']}, {ds_products[i].attrs['unit']}")
    unique = np.unique(ds_products[i].data.values)
    print(unique)
