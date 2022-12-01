import pathlib
import xarray as xr
import matplotlib.pyplot as plt

from lte_toolbox.mch.archiving.metadata import METADATA
from nowproject.visualization.plot_precip import plot_single_precip
from lte_toolbox.mch.visualization.plot import plot_precip_field

root_dir_path = pathlib.Path("/ltenas3/data/NowProject/snippet_mch/")

azc_zarr_dir_path = root_dir_path / "AZC" / "zarr"
ds_AZC = xr.open_zarr(azc_zarr_dir_path / "chunked_by_time.zarr")
plt.imshow(ds_AZC.data.isel(time=0))

time = ds_AZC.time.values[0]
time_str = str(time.astype('datetime64[s]'))
title = "AZC, Time: {}".format(time_str)
# ax, p = plot_single_precip(ds_AZC.data.isel(time=0), geodata=METADATA,
#                            title=title, figsize=(8, 5))
ax, p = plot_precip_field(ds_AZC.data.isel(time=0), title=title,
                          figsize=(8, 5))
