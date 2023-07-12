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
        self.raster_paths = None
        self.csv_path = None
        self.output_dir = None

        self.select_shapefile_button = tk.Button(master, text="Select Shapefile", command=self.select_shapefile)
        self.select_shapefile_button.pack()

        self.select_raster_button = tk.Button(master, text="Select Rasters", command=self.select_raster)
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
        self.raster_paths = filedialog.askopenfilenames(filetypes=(("GeoTIFF", "*.tif"), ("All files", "*.*")))

    def select_csv(self):
        self.csv_path = filedialog.askopenfilename(filetypes=(("CSV Files", "*.csv"), ("All files", "*.*")))

    def select_output_dir(self):
        self.output_dir = filedialog.askdirectory()

    def run(self):
        if self.shapefile_path is None or self.output_dir is None or self.raster_paths is None or self.csv_path is None:
            print("Please select the shapefile, rasters, CSV and output directory before running.")
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

        # load rasters and reproject to match intersection
        for raster_path in self.raster_paths:
            with rasterio.open(raster_path) as src:
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
                raster_column_name = os.path.splitext(os.path.basename(raster_path))[0]
                intersection[raster_column_name] = [stat['mean'] for stat in zonal_stats]

                # find maximum raster value
                max_raster_value = np.nanmax(reprojected_raster)

                # replace mean values greater than maximum raster value with NaN
                intersection.loc[intersection[raster_column_name] > max_raster_value, raster_column_name] = np.nan

        # load csv as GeoDataFrame
        df = pd.read_csv(self.csv_path)
        gdf_points = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.LONGNUM, df.LATNUM))
        gdf_points.set_crs("EPSG:4326", inplace=True)  # set CRS
        gdf_points = gdf_points.to_crs(epsg=3857)  # convert to Web Mercator projection

        # spatial join points with intersection
        joined = sjoin(gdf_points, intersection, how='inner')

        # calculate mean of observations for each grid square
        for column in df.columns:
            if column not in ['LATNUM', 'LONGNUM']:
                mean_obs = joined.groupby('index_right')[column].mean()
                intersection[column] = mean_obs

        # save intersection shapefile
        intersection.to_file(os.path.join(self.output_dir, 'intersection.shp'))

        # plot intersection with logged zonal stats data
        for column in [os.path.splitext(os.path.basename(raster_path))[0] for raster_path in self.raster_paths] + [col for col in df.columns if col not in ['LATNUM', 'LONGNUM']]:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            intersection.plot(column=np.log1p(intersection[column]), ax=ax, legend=True, edgecolor='black')
            plt.title(f'Logged {column}')
            plt.show()

root = tk.Tk()
app = App(root)
root.mainloop()
