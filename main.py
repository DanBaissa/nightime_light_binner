import os
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio import features
from rasterio.mask import mask
from shapely.geometry import shape, Polygon
import geopandas as gpd
from rasterio.plot import show
import tkinter as tk
from tkinter import filedialog

class App:
    def __init__(self, master):
        self.master = master
        master.title("Raster Processing")

        self.label = tk.Label(master, text="Raster Processing App")
        self.label.pack()

        self.input_dir_button = tk.Button(master, text="Select Raster Directory", command=self.select_input_dir)
        self.input_dir_button.pack()

        self.output_dir_button = tk.Button(master, text="Select Shapefile Output Directory", command=self.select_output_dir)
        self.output_dir_button.pack()

        self.run_button = tk.Button(master, text="Run", command=self.run)
        self.run_button.pack()

        self.input_dir = None
        self.output_dir = None

    def select_input_dir(self):
        self.input_dir = filedialog.askdirectory(title="Select directory containing raster files (.tif)")
        print(f"Selected input directory: {self.input_dir}")

    def select_output_dir(self):
        self.output_dir = filedialog.askdirectory(title="Select output directory for shapefile")
        print(f"Selected output directory: {self.output_dir}")

    def run(self):
        if self.input_dir is None or self.output_dir is None:
            print("Please select both the input and output directories before running.")
            return

        # get list of raster files
        raster_files = [os.path.join(self.input_dir, f) for f in os.listdir(self.input_dir) if f.endswith('.tif')]

        # read raster file
        r = rasterio.open(raster_files[0])  # assuming there's at least one .tif file

        # Creating Binning Function
        ppside = 100
        h = np.ceil(r.width / ppside)
        v = np.ceil(r.height / ppside)

        # create a transformation matrix for the aggregated raster
        transform = rasterio.transform.from_origin(
            r.bounds.left, r.bounds.top, r.transform[0] * h, r.transform[4] * v)

        # aggregate raster
        agg = r.read(1, out_shape=(int(v), int(h)))

        # convert aggregated raster to polygons
        agg_poly = features.shapes(agg.astype(np.int16), transform=transform)
        agg_poly = [shape(geom) for geom, val in agg_poly if val != 0]

        # create a GeoDataFrame from the polygons
        gdf = gpd.GeoDataFrame({'geometry': agg_poly})

        # calculate mean raster value for each polygon
        r_mean = []
        for geom in agg_poly:
            out_image, _ = mask(r, [geom], crop=True)
            r_mean.append(np.nanmean(out_image))

        # add the mean values to the GeoDataFrame
        gdf['BM_month_1'] = r_mean

        # Plotting the original raster with the grid overlay
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        show(r, ax=ax)
        gdf.boundary.plot(ax=ax, color='red')

        plt.show()

        # Plotting the binned night lights data
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        gdf.plot(column='BM_month_1', ax=ax, legend=True)

        plt.show()

        # write GeoDataFrame to shapefile
        output_filepath = os.path.join(self.output_dir, "Eth_Merged.shp")
        gdf.to_file(output_filepath)

root = tk.Tk()
app = App(root)
root.mainloop()
