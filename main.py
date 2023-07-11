import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
import rasterstats
import tkinter as tk
from tkinter import filedialog
from shapely.geometry import Polygon

class App:
    def __init__(self, master):
        self.master = master
        master.title("Shapefile Processing")

        self.label = tk.Label(master, text="Shapefile Processing App")
        self.label.pack()

        self.shapefile_button = tk.Button(master, text="Select Shapefile", command=self.select_shapefile)
        self.shapefile_button.pack()

        self.raster_button = tk.Button(master, text="Select Raster", command=self.select_raster)
        self.raster_button.pack()

        self.output_dir_button = tk.Button(master, text="Select Output Directory", command=self.select_output_dir)
        self.output_dir_button.pack()

        self.run_button = tk.Button(master, text="Run", command=self.run)
        self.run_button.pack()

        self.shapefile_path = None
        self.raster_path = None
        self.output_dir = None

    def select_shapefile(self):
        self.shapefile_path = filedialog.askopenfilename(title="Select Shapefile", filetypes=[("Shapefile", "*.shp")])
        print(f"Selected Shapefile: {self.shapefile_path}")

    def select_raster(self):
        self.raster_path = filedialog.askopenfilename(title="Select Raster", filetypes=[("Raster", "*.tif")])
        print(f"Selected Raster: {self.raster_path}")

    def select_output_dir(self):
        self.output_dir = filedialog.askdirectory(title="Select output directory")
        print(f"Selected output directory: {self.output_dir}")

    def run(self):
        if self.shapefile_path is None or self.output_dir is None or self.raster_path is None:
            print("Please select the shapefile, raster and output directory before running.")
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

        # save intersection to new shapefile
        intersection.to_file(f"{self.output_dir}/intersection.shp")

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

        # add zonal stats to intersection GeoDataFrame and plot
        intersection['mean'] = [stat['mean'] for stat in zonal_stats]

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))  # create a new figure and axes
        intersection.plot(column='mean', legend=True, ax=ax)
        plt.show()
        plt.close(fig)  # explicitly close the figure

root = tk.Tk()
app = App(root)
root.mainloop()
