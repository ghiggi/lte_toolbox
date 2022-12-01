#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 15:22:01 2022

@author: ghiggi
"""
from typing import Tuple, Union

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Normalize, Colormap
from matplotlib.image import AxesImage

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.geoaxes import GeoAxesSubplot
from pysteps.visualization.precipfields import get_colormap
# import pyproj
# from pysteps.visualization.utils import proj4_to_cartopy


PRECIP_VALID_TYPES = ("intensity", "depth", "prob")
PRECIP_VALID_UNITS = ("mm/h", "mm", "dBZ")


def _plot_map_cartopy(
    crs: ccrs.Projection,
    figsize: tuple = (8, 5),
    cartopy_scale: str = "50m",
    ax: Axes = None,
    drawlonlatlines: bool = False,
    drawlonlatlabels: bool = True,
    lw: float = 0.5
):
    """Plot coastlines, countries, rivers and meridians/parallels using cartopy.

    Parameters
    ----------
    crs : ccrs.Projection
        Instance of a crs class defined in cartopy.crs.
        It can be created using utils.proj4_to_cartopy.
    figsize : tuple, optional
        Figure size if ax is not specified, by default (8, 5)
    cartopy_scale : str, optional
        The scale (resolution) of the map. The available options are '10m',
        '50m', and '110m', by default "50m"
    ax : Axes, optional
        Axes object to plot the map on, by default None
    drawlonlatlines : bool, optional
        If true, plot longitudes and latitudes, by default False
    drawlonlatlabels : bool, optional
        If true, draw longitude and latitude labels. Valid only if
        'drawlonlatlines' is true, by default True
    lw : float, optional
        Line width, by default 0.5
    Returns
    -------
    Axes
        Cartopy axes. Compatible with matplotlib.
    """
    valid_cartopy_scale = ["10m", "50m", "110m"]
    if cartopy_scale not in valid_cartopy_scale:
        raise ValueError(f'Valid cartopy_scale are {valid_cartopy_scale}')

    if not ax:
        ax = plt.gca(figsize=figsize)

    if not isinstance(ax, GeoAxesSubplot):
        ax = plt.subplot(ax.get_subplotspec(), projection=crs)
        ax.set_axis_off()

    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "physical",
            "ocean",
            scale=cartopy_scale,
            edgecolor="none",
            facecolor=np.array([0.59375, 0.71484375, 0.8828125]),
        ),
        zorder=0,
    )
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "physical",
            "land",
            scale=cartopy_scale,
            edgecolor="none",
            facecolor=np.array([0.9375, 0.9375, 0.859375]),
        ),
        zorder=0,
    )
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "physical",
            "coastline",
            scale=cartopy_scale,
            edgecolor="black",
            facecolor="none",
            linewidth=lw,
        ),
        zorder=15,
    )
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "physical",
            "lakes",
            scale=cartopy_scale,
            edgecolor="none",
            facecolor=np.array([0.59375, 0.71484375, 0.8828125]),
        ),
        zorder=0,
    )
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "physical",
            "rivers_lake_centerlines",
            scale=cartopy_scale,
            edgecolor=np.array([0.59375, 0.71484375, 0.8828125]),
            facecolor="none",
        ),
        zorder=0,
    )
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "cultural",
            "admin_0_boundary_lines_land",
            scale=cartopy_scale,
            edgecolor="black",
            facecolor="none",
            linewidth=lw,
        ),
        zorder=15,
    )
    if cartopy_scale in ["10m", "50m"]:
        ax.add_feature(
            cfeature.NaturalEarthFeature(
                "physical",
                "reefs",
                scale="10m",
                edgecolor="black",
                facecolor="none",
                linewidth=lw,
            ),
            zorder=15,
        )
        ax.add_feature(
            cfeature.NaturalEarthFeature(
                "physical",
                "minor_islands",
                scale="10m",
                edgecolor="black",
                facecolor="none",
                linewidth=lw,
            ),
            zorder=15,
        )

    if drawlonlatlines:
        grid_lines = ax.gridlines(
            crs=ccrs.PlateCarree(), draw_labels=drawlonlatlabels, dms=True
        )
        grid_lines.top_labels = grid_lines.right_labels = False
        grid_lines.y_inline = grid_lines.x_inline = False
        grid_lines.rotate_labels = False

    return ax


def get_cbar_label(ptype, probthr, units):
    if ptype == "intensity":
        cbar_label = f"Precipitation intensity ({units})"
    elif ptype == "depth":
        cbar_label = f"Precipitation depth ({units})"
    else:
        cbar_label = f"P(R > {probthr:.1f} {units})"
    return cbar_label


def plot_precip_field(da: xr.DataArray,
                      ax: Axes = None,
                      ptype="intensity",
                      units="mm/h",
                      colorscale="pysteps",
                      figsize=(8, 5),
                      title: str = None,
                      colorbar: bool = True,
                      norm: Normalize = None,
                      cmap: Union[Colormap, str] = None,
                      drawlonlatlines: bool = False,
                      extent: Tuple[Union[int, float]] = None,
                      probthr: float = None,
                      ) -> Tuple[GeoAxesSubplot, AxesImage]:
    """Plot a single precipitation event (one timestep).
    Parameters
    ----------
    da : xr.DataArray
        DataArray containing the data to plot, with two
        spatial dimensions (y-x, lat-lon)
    ax : Axes, optional
        Axes object to plot the map on, by default None
    ptype : str, optional
        Type of the map to plot. Options : {'intensity',
        'depth', 'prob'}, by default "intensity"
    units : str, optional
        Units of the input array. Options : {'mm/h',
        'mm', 'dBZ'}, by default "mm/h"
    colorscale : str, optional
        Colorscale to use. Options : {'pysteps', 'STEPS-BE',
        'BOM-RF3'}, by default "pysteps"
    colorbar : bool, optional
        If true, add colorbar on the right side of the plot,
        by default True
    figsize : tuple, optional
        Figure size if ax is not specified, by default (8,5)
    title : str, optional
        If not None, print the title on top of the plot,
        by default None

    drawlonlatlines : bool, optional
        If true, plot longitudes and latitudes, by default False
    extent : Tuple[Union[int,float]], optional
        bounding box in data coordinates that the image will fill,
        by default None
    probthr : float, optional
        Intensity threshold to show in the color bar of the
        exceedance probability map. Required if ptype is “prob”
        and colorbar is True, by default None
    norm : Normalize, optional
        Normalize instance used to scale scalar data to the [0, 1]
        range before mapping to colors using cmap, by default None
    cmap : Union[Colormap, str], optional
        Colormap instance or registered colormap name used to map
        scalar data to colors, by default None
    Returns
    -------
    Tuple[GeoAxesSubplot, AxesImage]
        The subplot
    """
    if not isinstance(da, xr.DataArray):
        raise TypeError("plot_precip_fields expects a xr.DataArray.")

    # Define colorbar appearance
    cbar_label = get_cbar_label(ptype, probthr, units)
    if not cmap and not norm:
        cmap, norm, clevs, clevs_str = get_colormap(ptype, units, colorscale)
        cbar_kwargs = {
            "ticks": clevs,
            "spacing": "uniform",
            "extend": "max",
            "shrink": 0.8,
        }
    else:
        clevs_str = None
        cbar_kwargs = {}

    # Get crs
    crs_ref = ccrs.epsg(21781)
    # crs_ref = proj4_to_cartopy(pyproj.CRS.from_epsg(21781))
    crs_proj = crs_ref

    # Create axis if not provided
    if ax is None:
        _, ax = plt.subplots(
                figsize=figsize,
                subplot_kw={'projection': crs_proj}
            )

    # Display the image
    p = da.plot.imshow(
        ax=ax,
        transform=crs_ref,
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        add_colorbar=colorbar,
        cbar_kwargs=cbar_kwargs if colorbar else None,
        zorder=1,
    )

    # Set title
    if title:
        ax.set_title(title)

    # Edit colorbar
    if colorbar:
        if clevs_str is not None:
            p.colorbar.ax.set_yticklabels(clevs_str)
        p.colorbar.set_label(cbar_label)

    # Add background
    p.axes = _plot_map_cartopy(crs_proj,
                               cartopy_scale="50m",
                               drawlonlatlines=drawlonlatlines,
                               ax=p.axes)

    # Restrict the extent
    if extent:
        p.axes.set_extent(extent, crs_ref)

    return ax, p


def plot_precip_fields(da: xr.DataArray,
                       col,
                       col_wrap,
                       ptype="intensity",
                       units="mm/h",
                       colorscale="pysteps",
                       figsize=(12,8),
                       title: str = None,
                       colorbar: bool = True,
                       probthr: float = None,
                       drawlonlatlines: bool = False,
                       extent: Tuple[Union[int,float]] = None,
                       # norm: Normalize = None,
                       # cmap: Union[Colormap, str] = None
                       ):
    """Plot a single precipitation event (one timestep).
    Parameters
    ----------
    da : xr.DataArray
        DataArray containing the data to plot, with two
        spatial dimensions (y-x, lat-lon)

    ptype : str, optional
        Type of the map to plot. Options : {'intensity',
        'depth', 'prob'}, by default "intensity"
    units : str, optional
        Units of the input array. Options : {'mm/h',
        'mm', 'dBZ'}, by default "mm/h"
    colorscale : str, optional
        Colorscale to use. Options : {'pysteps', 'STEPS-BE',
        'BOM-RF3'}, by default "pysteps"
    colorbar : bool, optional
        If true, add colorbar on the right side of the plot,
        by default True
    figsize : tuple, optional
        Figure size if ax is not specified, by default (8,5)
    title : str, optional
        If not None, print the title on top of the plot,
        by default None

    drawlonlatlines : bool, optional
        If true, plot longitudes and latitudes, by default False
    extent : Tuple[Union[int,float]], optional
        bounding box in data coordinates that the image will fill,
        by default None
    probthr : float, optional
        Intensity threshold to show in the color bar of the
        exceedance probability map. Required if ptype is “prob”
        and colorbar is True, by default None
    norm : Normalize, optional
        Normalize instance used to scale scalar data to the [0, 1]
        range before mapping to colors using cmap, by default None
    cmap : Union[Colormap, str], optional
        Colormap instance or registered colormap name used to map
        scalar data to colors, by default None
    Returns
    -------
    Tuple[GeoAxesSubplot, AxesImage]
        The subplot
    """
    if not isinstance(da, xr.DataArray):
        raise TypeError("plot_precip_fields expects a xr.DataArray.")
    # Define colorbar appearance
    cmap, norm, clevs, clevs_str = get_colormap(ptype, units, colorscale)
    cbar_label = get_cbar_label(ptype, probthr, units)
    cbar_kwargs = {
        "ticks": clevs,
        "spacing": "uniform",
        "extend": "max",
        "shrink": 0.8,
        "label": cbar_label
    }

    # Get crs
    crs_ref = ccrs.epsg(21781)
    # crs_ref = proj4_to_cartopy(pyproj.CRS.from_epsg(21781))
    crs_proj = crs_ref

    # Plot image
    p = da.plot.imshow(
        subplot_kws={'projection': crs_proj},
        figsize=figsize,
        transform=crs_ref,
        col=col, col_wrap=col_wrap,
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        add_colorbar=colorbar,
        cbar_kwargs=cbar_kwargs,
        zorder=1,
    )

    # Edit colorbar
    # if colorbar:
    #     if clevs_str is not None:
    #         p.colorbar.ax.set_yticklabels(clevs_str)
    #     p.colorbar.set_label(cbar_label)

    # Set title
    if title:
        p.fig.suptitle(title, x=0.4, y=1.0, fontsize="x-large")

    # Add background and set extent
    for ax in p.axes.flatten():
        ax = _plot_map_cartopy(crs_proj,
                               cartopy_scale="50m",
                               drawlonlatlines=drawlonlatlines,
                               ax=ax)

        if extent:
            ax.set_extent(extent, crs_ref)

    return p


