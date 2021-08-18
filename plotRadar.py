#!/usr/bin/env python3
# Next-gen HDWX radar plotting script
# Created 7 July 2021 by Sam Gardner <stgardner4@tamu.edu>

from datetime import datetime as dt
import pyart
from matplotlib import pyplot as plt
from os import path, getcwd, listdir
from cartopy import crs as ccrs
from metpy.plots import ctables
from metpy.plots import USCOUNTIES
import numpy as np
import warnings
import multiprocessing as mp
from matplotlib import image as mpimage

def plot_ppi_map_modified(
            rmd, field, sweep=0, mask_tuple=None,
            vmin=None, vmax=None, cmap=None, norm=None, mask_outside=False,
            title=None, title_flag=True,
            colorbar_flag=True, colorbar_label=None, ax=None, fig=None,
            lat_lines=None, lon_lines=None, projection=None,
            min_lon=None, max_lon=None, min_lat=None, max_lat=None,
            width=None, height=None, lon_0=None, lat_0=None,
            resolution='110m', shapefile=None, shapefile_kwargs=None,
            edges=True, gatefilter=None,
            filter_transitions=True, embelish=True, raster=False,
            ticks=None, ticklabs=None, alpha=None):
        # parse parameters
        ax, fig = pyart.graph.common.parse_ax_fig(ax, fig)
        vmin, vmax = pyart.graph.common.parse_vmin_vmax(rmd._radar, field, vmin, vmax)
        cmap = pyart.graph.common.parse_cmap(cmap, field)
        if lat_lines is None:
            lat_lines = np.arange(30, 46, 1)
        if lon_lines is None:
            lon_lines = np.arange(-110, -75, 1)
        lat_0 = rmd.loc[0]
        lon_0 = rmd.loc[1]

        # get data for the plot
        data = rmd._get_data(
            field, sweep, mask_tuple, filter_transitions, gatefilter)
        x, y = rmd._get_x_y(sweep, edges, filter_transitions)

        # mask the data where outside the limits
        if mask_outside:
            data = np.ma.masked_outside(data, vmin, vmax)

        # initialize instance of GeoAxes if not provided
        if hasattr(ax, 'projection'):
            projection = ax.projection
        else:
            if projection is None:
                # set map projection to LambertConformal if none is specified
                projection = ccrs.LambertConformal(
                    central_longitude=lon_0, central_latitude=lat_0)
                warnings.warn("No projection was defined for the axes."
                              + " Overridding defined axes and using default "
                              + "axes.", UserWarning)
            ax = plt.axes(projection=projection)

        if min_lon:
            ax.set_extent([min_lon, max_lon, min_lat, max_lat],
                          crs=ccrs.PlateCarree())
        elif width:
            ax.set_extent([-width/2., width/2., -height/2., height/2.],
                          crs=rmd.grid_projection)

        # plot the data
        if norm is not None:  # if norm is set do not override with vmin/vmax
            vmin = vmax = None
        pm = ax.pcolormesh(x * 1000., y * 1000., data, alpha=alpha,
                           vmin=vmin, vmax=vmax, cmap=cmap,
                           norm=norm, transform=rmd.grid_projection)

        # plot as raster in vector graphics files
        if raster:
            pm.set_rasterized(True)

        if title_flag:
            rmd._set_title(field, sweep, title, ax)

        # add plot and field to lists
        rmd.plots.append(pm)
        rmd.plot_vars.append(field)

        if colorbar_flag:
            rmd.plot_colorbar(
                mappable=pm, label=colorbar_label, field=field, fig=fig,
                ax=ax, ticks=ticks, ticklabs=ticklabs)
        # keep track of this GeoAxes object for later
        rmd.ax = ax
        return pm

def plot_radar(radarFileName):
    px = 1/plt.rcParams["figure.dpi"]
    basePath = path.join(getcwd(), "output")
    radarDataDir = path.join(getcwd(), "radarData")
    radarFilePath = path.join(radarDataDir, radarFileName)
    try:
        radar = pyart.io.read(radarFilePath)
    except Exception as e:
        warningString = str(dt.utcnow())+" error reading "+radarFileName+": "+str(e)+"\n"
        logFile = open("warnings.log", "a")
        logFile.write(warningString)
        logFile.close()
        return
    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()})
    fig.set_size_inches(1920*px, 1080*px)
    norm, cmap = ctables.registry.get_with_steps("NWSReflectivity", 5, 5)
    cmap.set_under("#00000000")
    cmap.set_over("black")
    ADRADMapDisplay = pyart.graph.RadarMapDisplay(radar)
    plotHandle = plot_ppi_map_modified(ADRADMapDisplay, "reflectivity", 0, resolution="10m", embelish=False, cmap=cmap, norm=norm, colorbar_flag=False)
    ADRADMapDisplay.set_aspect_ratio(1)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        ADRADMapDisplay.plot_range_rings([40, 80, 120, 160], col="gray", ls="dotted")
    ax.add_feature(USCOUNTIES.with_scale("5m"), edgecolor="gray")
    infoString = str()
    if "instrument_name" in radar.metadata.keys():
        infoString = radar.metadata["instrument_name"].decode()
    if "sigmet_task_name" in radar.metadata.keys():
        infoString = infoString + " " +radar.metadata["sigmet_task_name"].decode().replace("  ", "")
    infoString = infoString + " PPI\n"
    if "prt" in radar.instrument_parameters:
        prf = np.round(1/np.mean(radar.instrument_parameters["prt"]["data"]), 0)
        infoString = infoString + "Avg. PRF: " + str(prf) + " Hz"
    elevation = np.round(radar.fixed_angle["data"][0], 1)
    infoString = infoString + "    Elevation: " + str(elevation) + "°"
    if "unambiguous_range" in radar.instrument_parameters:
        maxRange = np.round(np.max(radar.instrument_parameters["unambiguous_range"]["data"])/1000, 0)
        infoString = infoString + "    Max Range: " + str(maxRange) + " km\n"
    infoString = infoString + pyart.util.datetime_from_radar(radar).strftime("%d %b %Y %H:%M:%S UTC")
    ax.set_title(infoString)
    cbax = fig.add_axes([ax.get_position().x0, 0.075, (ax.get_position().width/3), .02])
    fig.colorbar(plotHandle, cax=cbax, orientation="horizontal", extend="neither")
    cbax.set_xlabel("Reflectivity (dBZ)")
    lax = fig.add_axes([ax.get_position().x0+2*(ax.get_position().width/3), 0.015, (ax.get_position().width/3), .1])
    lax.set_aspect(2821/11071)
    lax.axis("off")
    plt.setp(lax.spines.values(), visible=False)
    atmoLogo = mpimage.imread("assets/atmoLogo.png")
    lax.imshow(atmoLogo)
    fig.savefig(path.join(basePath, str(sorted(listdir(radarDataDir)).index(radarFileName))+".png"), bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    radarDataDir = path.join(getcwd(), "radarData")
    with mp.Pool(processes=12) as pool:
        pool.map(plot_radar, sorted(listdir(radarDataDir)))
        