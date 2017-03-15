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



# Get the region
regions = scipy.spatial.Delaunay(coords)
edges = utils.get_edges_from_triangles(regions.simplices)

depth_map, focus_meas = utils.cal_depth_map(imgs, coords)

# make samples

pyplot.figure('triangles')
pyplot.triplot(coords[:, 0], coords[:, 1], regions.simplices)
pyplot.figure('pts')
x, y = coords[0]
pyplot.plot(x, y, 'o')
for x, y in coords[1:]:
    pyplot.plot(x, y, 'x')
pyplot.show()

