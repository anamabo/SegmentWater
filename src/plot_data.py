import rasterio
from matplotlib import pyplot

dataset = rasterio.open("long-122.4194_lat37.7749.tif")

pyplot.imshow(dataset.read(1), cmap='pink')
pyplot.show()
