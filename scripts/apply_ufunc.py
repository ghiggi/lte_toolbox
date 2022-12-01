import pathlib
import xarray as xr

root_dir_path = pathlib.Path("/ltenas3/data/NowProject/snippet_mch/")

azc_zarr_dir_path = root_dir_path / "AZC" / "zarr"
ds_AZC = xr.open_zarr(azc_zarr_dir_path / "chunked_by_time.zarr")

ds_AZC.data.mean(dim="time")