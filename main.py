import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.io import MemoryFile
import rasterstats
import tkinter as tk
from tkinter import filedialog
from shapely.geometry import Polygon
import os
import pandas as pd
from shapely.geometry import Point
from geopandas.tools import sjoin

class App:
    def __init__(self, master):
        self.master = master
        master.title("Raster Merger")

        self.shapefile_path = None
        self.raster_path = None
        self.csv_path = None
        self.output_dir = None

        self.select_shapefile_button = tk.Button(master, text="Select Shapefile", command=self.select_shapefile)
        self.select_shapefile_button.pack()

        self.select_raster_button = tk.Button(master, text="Select Raster", command=self.select_raster)
        self.select_raster_button.pack()

        self.select_csv_button = tk.Button(master, text="Select CSV", command=self.select_csv)
        self.select_csv_button.pack()

        self.select_output_dir_button = tk.Button(master, text="Select Output Directory", command=self.select_output_dir)
        self.select_output_dir_button.pack()

        self.run_button = tk.Button(master, text="Run", command=self.run)
        self.run_button.pack()

    def select_shapefile(self):
        self.shapefile_path = filedialog.askopenfilename(filetypes=(("Shapefile", "*.shp"), ("All files", "*.*")))

    def select_raster(self):
        self.raster_path = filedialog.askopenfilename(filetypes=(("GeoTIFF", "*.tif"), ("All files", "*.*")))

    def select_csv(self):
        self.csv_path = filedialog.askopenfilename(filetypes=(("CSV Files", "*.csv"), ("All files", "*.*")))

    def select_output_dir(self):
        self.output_dir = filedialog.askdirectory()

    def run(self):
        if self.shapefile_path is None or self.output_dir is None or self.raster_path is None or self.csv_path is None:
            print("Please select the shapefile, raster, CSV and output directory before running.")
            return

        # load shapefile
        gdf = gpd.read_file(self.shapefile_path)

        # set CRS if not already set
        if gdf.crs is None:
            gdf.set_crs("EPSG:4326", inplace=True)

        # convert to Web Mercator projection (units in meters)
        gdf = gdf.to_crs(epsg=3857)

        # create grid
        xmin, ymin, xmax, ymax = gdf.total_bounds
        width = height = 10_000  # 10km grid size

        rows = int(np.ceil((ymax-ymin) / height))
        cols = int(np.ceil((xmax-xmin) / width))

        polygons = []
        for x in range(cols):
            for y in range(rows):
                polygons.append(Polygon([(x*width+xmin, y*height+ymin),
                                         ((x+1)*width+xmin, y*height+ymin),
                                         ((x+1)*width+xmin, (y+1)*height+ymin),
                                         (x*width+xmin, (y+1)*height+ymin)]))

        grid = gpd.GeoDataFrame({'geometry': polygons}, crs='EPSG:3857')

        # perform intersection between shapefile and grid
        intersection = gpd.overlay(gdf, grid, how='intersection')

        # plot intersection
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        intersection.plot(ax=ax, edgecolor='black', facecolor='white')
        plt.show()
        plt.close(fig)  # explicitly close the figure

        # load raster and reproject to match intersection
        with rasterio.open(self.raster_path) as src:
            transform, width, height = calculate_default_transform(src.crs, intersection.crs.to_string(), src.width, src.height, *src.bounds)
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': intersection.crs.to_string(),
                'transform': transform,
                'width': width,
                'height': height
            })

            with MemoryFile() as memfile:
                with memfile.open(**kwargs) as dest:
                    reproject(
                        source=rasterio.band(src, 1),
                        destination=rasterio.band(dest, 1),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=intersection.crs.to_string(),
                        resampling=Resampling.nearest
                    )
                reprojected_raster = memfile.open().read(1)

        # calculate zonal statistics
        zonal_stats = rasterstats.zonal_stats(intersection, reprojected_raster, affine=kwargs['transform'], stats=['mean'])

        # add zonal stats to intersection GeoDataFrame
        raster_column_name = os.path.splitext(os.path.basename(self.raster_path))[0]
        intersection[raster_column_name] = [stat['mean'] for stat in zonal_stats]

        # find maximum raster value
        max_raster_value = np.nanmax(reprojected_raster)

        # replace mean values greater than maximum raster value with NaN
        intersection.loc[intersection[raster_column_name] > max_raster_value, raster_column_name] = np.nan

        # load csv as GeoDataFrame
        df = pd.read_csv(self.csv_path)
        gdf_points = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.LONGNUM, df.LATNUM), crs='EPSG:4326')
        gdf_points = gdf_points.to_crs(intersection.crs)

        # spatial join between grid and points
        joined = sjoin(gdf_points, intersection, how='inner', op='within')

        # check if the spatial join operation was successful
        if 'index_right' not in joined.columns:
            print("No points were found within the polygons.")
            return

        # calculate mean of observations for each grid square for each column
        for column in ['All_Population_Count_2000', 'All_Population_Count_2005', 'All_Population_Count_2010',
                       'All_Population_Count_2015', 'All_Population_Count_2020']:
            mean_obs = joined.groupby('index_right')[column].mean()
            intersection[f'mean_{column}'] = mean_obs

        # save intersection to new shapefile
        intersection.to_file(f"{self.output_dir}/intersection.shp")

        # plot intersection with logged zonal stats data
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        intersection.plot(column=np.log1p(intersection[raster_column_name]), ax=ax, legend=True, edgecolor='black')
        plt.show()

root = tk.Tk()
app = App(root)
root.mainloop()
