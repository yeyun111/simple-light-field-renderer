import os
import numpy
import scipy.spatial
import cv2
from matplotlib import pyplot
import utils

DEFAULT_NUM_INTERP = 100

input_path = r'examples/Batman'

# Preprocess a image list
img_list = [os.sep.join([input_path, x]) for x in os.listdir(input_path) if x.lower().endswith('jpg')]
imgs = [utils.limit_image_size(cv2.imread(x)) for x in img_list]

# Calibrate images
imgs, coords = utils.calibrate_images(imgs)

x_min = numpy.min(coords[:, 0])
x_max = numpy.max(coords[:, 0])
y_min = numpy.min(coords[:, 1])
y_max = numpy.max(coords[:, 1])

w = x_max - x_min
h = y_max - y_min

# Get the region
regions = scipy.spatial.Delaunay(coords)
edges = utils.get_edges_from_triangles(regions.simplices)

qhull = scipy.spatial.ConvexHull(regions.points)
area_ratio = qhull.volume / w / h

vol_samples = DEFAULT_NUM_INTERP / area_ratio

num_x_samples = int(numpy.sqrt(vol_samples * w / h) + 0.5)
num_y_samples = int(numpy.sqrt(vol_samples * w / h) * h / w + 0.5)

depth_map, focus_meas = utils.cal_depth_map(imgs, coords)

pyplot.figure('depth')
pyplot.imshow(depth_map, cmap='gray')

# make samples
x_samples = numpy.linspace(x_min, x_max, num_x_samples + 2)
y_samples = numpy.linspace(y_min, y_max, num_y_samples + 2)

# interpolate with 3 images around, to avoid error form far images and reduce computations
for x in x_samples:
    for y in y_samples:
        for triangle in regions.simplices:
            if inside((x, y), triangle)

pyplot.figure('triangles')
pyplot.triplot(coords[:, 0], coords[:, 1], regions.simplices)
pyplot.figure('pts')
x, y = coords[0]
pyplot.plot(x, y, 'o')
for x, y in coords[1:]:
    pyplot.plot(x, y, 'x')
pyplot.show()
