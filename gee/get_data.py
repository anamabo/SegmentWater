import os
import time

import ee
from oauth2client.service_account import ServiceAccountCredentials
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive


def get_sentinel_image(
    longitude: float,
    latitude: float,
    tile_size: int,
    init_date: str,
    end_date: str,
):
    center = ee.Geometry.Point([longitude, latitude])

    # Create a bbox of a buffer around the center point
    tile = center.buffer(tile_size / 2).bounds()

    # Load the Sentinel-2 image collection.
    # FilterBounds selects the sentinel tile intersecting the tile
    # FilterDate filters the images outside the date range
    # Filter.lt Filter images with more than 10% cloud coverage
    # Take the median to obtain a final image.
    sentinel2_image = (
        ee.ImageCollection("COPERNICUS/S2")
        .filterBounds(tile)
        .filterDate(init_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10))
        .median()
    )

    # Select the desired bands (e.g., B4, B3, B2 for true color)
    sentinel2_image_rgb = sentinel2_image.select(["B4", "B3", "B2"])

    # clipped image to the tile size
    sentinel2_image_clipped = sentinel2_image_rgb.clip(tile)

    return tile, sentinel2_image_clipped


# TODO:
#  add cli args for the SA, credentials and project_id
service_account = "gee-961@genial-airway-429107-d4.iam.gserviceaccount.com"
sa_json_file = os.path.join(os.getcwd(), ".credentials_gee.json")
project_id = "genial-airway-429107-d4"
gdrive_folder = "PaliGemma_data"

credentials = ee.ServiceAccountCredentials(service_account, sa_json_file)
ee.Initialize(credentials, project=project_id)

longitude = -122.4194
latitude = 37.7749
tile_size = 2500  # tile size in meters
init_date = "2023-02-01"
end_date = "2023-03-31"
output_filename = f"long{longitude}_lat{latitude}"

image_tile, sentinel_image = get_sentinel_image(
    longitude=longitude,
    latitude=latitude,
    tile_size=tile_size,
    init_date=init_date,
    end_date=end_date,
)

projection = sentinel_image.select("B2").projection().getInfo()

# Export the image as a GeoTIFF file
export_task = ee.batch.Export.image.toDrive(
    image=sentinel_image,
    folder=gdrive_folder,
    description=output_filename,
    crs=projection["crs"],
    # crsTransform=projection['transform'],
    fileFormat="GeoTIFF",
    region=image_tile.getInfo()["coordinates"],
)
# export_task = ee.batch.Export.image.toCloudStorage(
#     image=sentinel_image,
#     description=f"long{longitude}_lat{latitude}",
#     bucket="paligemma_data",
#     crs=projection["crs"],
#     crsTransform=projection["transform"],
#     region=image_tile.getInfo()["coordinates"],
#     #fileFormat="GeoTIFF",
# )
export_task.start()

print("Exporting image...")

# Monitor the export task
while export_task.active():
    print("Polling for task (id: {}).".format(export_task.id))
    time.sleep(30)  # Check every 30 seconds

print("Export complete.")

# authenticate to Google Drive (of the Service account)
# The image is stored there!
gauth = GoogleAuth()
scopes = ["https://www.googleapis.com/auth/drive"]
gauth.credentials = ServiceAccountCredentials.from_json_keyfile_name(
    sa_json_file, scopes=scopes
)
drive = GoogleDrive(gauth)

# Downloading the images from Drive (of the Service account)
# We need to get the id of the folder where the images are saved.
file_list = drive.ListFile(
    {"q": "'root' in parents and trashed=false"}
).GetList()
file_dict = {
    f"{file['title']}": file["id"]
    for file in file_list
    if gdrive_folder in file["title"]
}

file_list1 = drive.ListFile(
    {"q": f" '{file_dict[gdrive_folder]}' in parents and trashed=false"}
).GetList()
for file in file_list1:
    filename = file["title"]
    print(f"Downloading {filename} to local...")
    file.GetContentFile(filename, mimetype="image/tiff")
    file.Delete()
