"""
This is the main pipeline of the project
"""
import ee
import os
import geemap
import time


def get_sentinel_image(
        longitude: float,
        latitude: float,
        tile_size: int,
        init_date: str,
        end_date: str,
):
    center = ee.Geometry.Point([longitude, latitude])

    # Create a buffer around the center point to get a square tile
    tile = center.buffer(tile_size / 2).bounds()

    # Load the Sentinel-2 image collection
    sentinel2 = ee.ImageCollection('COPERNICUS/S2') \
        .filterBounds(tile) \
        .filterDate(init_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)) \
        .median()  # Take the median to reduce cloud cover

    # Select the desired bands (e.g., B4, B3, B2 for true color)
    sentinel2_rgb = sentinel2.select(['B4', 'B3', 'B2'])
    return tile, sentinel2_rgb



service_account = "gee-961@genial-airway-429107-d4.iam.gserviceaccount.com"
sa_json_file = os.path.join(os.getcwd(), ".genial-airway-429107-d4-c315ddc9e712.json")
project_id = "genial-airway-429107-d4"
credentials = ee.ServiceAccountCredentials(service_account, sa_json_file)
ee.Initialize(credentials, project=project_id)

longitude = -122.4194
latitude = 37.7749
tile_size = 2500  # tile size in meters
init_date = "2023-02-01"
end_date = "2023-03-31"

image_tile, sentinel_image = get_sentinel_image(
    longitude=longitude,
    latitude=latitude,
    tile_size=tile_size,
    init_date=init_date,
    end_date=end_date,
)

clipped_image = sentinel_image.clip(image_tile)

# # Define visualization parameters
# vis_params = {
#     'min': 0,
#     'max': 3000,
#     'bands': ['B4', 'B3', 'B2']
# }
#
# # Display the tile on the map
# Map = geemap.Map()
# Map.centerObject(image_tile, zoom=15)
# Map.addLayer(clipped_image, vis_params, 'Sentinel-2 RGB')
# Map.addLayer(image_tile, {}, 'Tile Boundary')

# Export the image as a GeoTIFF file
export_task = ee.batch.Export.image.toDrive(
    image=clipped_image,
    folder="PaliGemma_data",
    description=f"{longitude}_{latitude}",  # to change!!!!
    scale=10,  # Sentinel-2 resolution is 10 meters
    region=image_tile.getInfo()["coordinates"],
    fileFormat='GeoTIFF',
    crs='EPSG:4326'
)
export_task.start()

print("Exporting image...")

# Monitor the export task
while export_task.active():
    print('Polling for task (id: {}).'.format(export_task.id))
    time.sleep(30)  # Check every 30 seconds

print("Export complete.")