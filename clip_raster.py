import gdal, os, rasterio
import numpy as np
from rasterio.plot import show
import geopandas as gpd
import glob
import fiona
from subprocess import Popen
import sys

def input_tile(tile_number):
    main_raster_filepath = "tiff_shp/" + str(tile_number)
    large_tiff_filepath = "large_tiff_images/" + str(tile_number) + ".tiff"
    grid_output = "tiff_shp/box_" + str(tile_number) + ".shp"

    return main_raster_filepath, large_tiff_filepath, grid_output


def large_tiff_to_shp(large_tiff_filepath):
    '''
    Transform main large raster tiff to shapefile

    Input:
    large_tiff_filepath: main tiff filepath

    Returns:
    raster_shp: raster shapefile

    '''
    im = large_tiff_filepath
    data = rasterio.open(im)

    bounds = data.bounds

    from shapely.geometry import box

    geom = box(*bounds)

    raster_shp = gpd.GeoDataFrame({"id":1,"geometry":[geom]})

    return raster_shp

def filter_gid_shp(raster_shp, grid_output):
    '''
    Filter unique id or gid inside main raster 

    Input:
    grid_output: output filtered grid 

    Returns:

    '''
    grid = gpd.read_file('grid_malang.shp')
    #grid['id'] = grid['id'].astype('int64')

    grid_sjoin = gpd.sjoin(grid, raster_shp, how='inner', op='within')
    grid_sjoin.to_file(grid_output)


def create_single_shp(grid_output, main_raster_filepath):
    '''
    Created single shp for each grid inside grid_sjoin

    Input:
    grid_output: output filtered grid 
    main_raster_id

    Returns:

    '''

    with fiona.open(grid_output, 'r') as dst_in:
        for index, feature in enumerate(dst_in):
            with fiona.open(main_raster_filepath + '/' + str(index + 1) +'.shp', 'w', **dst_in.meta) as dst_out:
                dst_out.write(feature)

def create_tiff(main_raster_filepath, large_tiff_filepath):

    '''
    Created tiff from each single shp

    Input:
    main_raster_id
    large_tiff_filepath: main tiff filepath

    Returns:

    '''
    if not os.path.exists('small_tiff_images/'):
        os.makedirs('small_tiff_images/')

    polygons = glob.glob(main_raster_filepath + '/' + '*.shp')  ## Retrieve all the .shp files

    for polygon in polygons:
        name = gpd.read_file(polygon)['gid_100'][0]
        command = 'gdalwarp -dstnodata -9999 -cutline {} ' \
                '-crop_to_cutline -of GTiff {} small_tiff_images/{}.tiff'.format(polygon, large_tiff_filepath, name)
        Popen(command, shell=True)

def main():
    """
    Loading the data from csv files, merging the data, removing duplicate data, and saving it in SQL database.
    
    """
    
    if len(sys.argv) == 2:
        
        tile_number = sys.argv[1:]
        tile_number = tile_number[0]
        
        main_raster_filepath = f'tiff_shp/{tile_number}'
        if not os.path.exists('tiff_shp/'):
            os.makedirs('tiff_shp/')
        
        large_tiff_filepath = f'large_tiff_images/{tile_number}.tiff'
        if not os.path.exists('large_tiff_images/'):
            os.makedirs('large_tiff_images/')
        
        grid_output = f'tiff_shp/box_{tile_number}.shp'
        
        #main_raster_filepath, large_tiff_filepath, grid_output = input_tile(tile_number)

        print('Create shp from the main raster...')
        raster_shp = large_tiff_to_shp(large_tiff_filepath)

        print('Filter grid inside the main raster...')
        filter_gid_shp(raster_shp, grid_output)

        print('Create single shp from each grid inside the main raster...')
        create_single_shp(grid_output, main_raster_filepath)

        print('Create tiff from each single shp...')
        create_tiff(main_raster_filepath, large_tiff_filepath)
        
        print('Finish!')
    
    else:
        print('Check again!')

if __name__ == '__main__':
    main()