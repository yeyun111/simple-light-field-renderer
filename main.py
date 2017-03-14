import os
import numpy
import scipy.spatial
import cv2
from matplotlib import pyplot
import utils

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

depth_map = utils.cal_depth_map(imgs, coords)

pyplot.imshow(depth_map, cmap='gray')
pyplot.show()

'''
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
gray_imgs = [cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in imgs]
depth_maps = []
for edge in edges:
    i0, i1 = edge
    disparity = stereo.compute(gray_imgs[i0], gray_imgs[i1]).astype(numpy.float32)
    depth_maps.append(disparity)
depth_maps = numpy.asarray(depth_maps)
depth_map = numpy.mean(depth_maps, axis=0)

pyplot.figure('depth map')
pyplot.imshow(depth_maps[0], 'gray')
pyplot.show()
'''

pyplot.figure('triangles')
pyplot.triplot(coords[:, 0], coords[:, 1], regions.simplices)
pyplot.figure('pts')
x, y = coords[0]
pyplot.plot(x, y, 'o')
for x, y in coords[1:]:
    pyplot.plot(x, y, 'x')
pyplot.show()
