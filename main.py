import os
import numpy
import scipy.spatial
import cv2
from matplotlib import pyplot
import utils

DEFAULT_NUM_INTERP = 10

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

qhull_polygon = numpy.array([qhull.points[i] for i in qhull.vertices], dtype=numpy.float32)

vol_samples = DEFAULT_NUM_INTERP / area_ratio

num_x_samples = int(numpy.sqrt(vol_samples * w / h) + 0.5)
num_y_samples = int(numpy.sqrt(vol_samples * w / h) * h / w + 0.5)

depth_map, focus_meas = utils.cal_depth_map(imgs, coords)

pyplot.figure('depth')
pyplot.imshow(depth_map, cmap='gray')

# make samples
x_interval = (x_max - x_min) / num_x_samples
y_interval = (y_max - y_min) / num_y_samples
x_samples = numpy.linspace(x_min + 0.5*x_interval, x_max - 0.5*x_interval, num_x_samples)
y_samples = numpy.linspace(y_min + 0.5*y_interval, y_max - 0.5*y_interval, num_y_samples)

interp_coords = {}

# interpolate with 3 images around, to avoid error form far images and reduce computations
triangles = numpy.sort(regions.simplices)
interp_coords = {tuple(x): [] for x in triangles}
for x in x_samples:
    for y in y_samples:
        if cv2.pointPolygonTest(qhull_polygon, (x, y), False) > 0:
            for triangle in triangles:
                polygon = numpy.array([regions.points[i] for i in triangle], dtype=numpy.float32)
                if cv2.pointPolygonTest(polygon, (x, y), False) > 0:
                    interp_coords[tuple(triangle)].append((x, y))

n_samples = sum([len(x) for x in interp_coords.values()])

located_images = [(coord, img) for (coord, img) in zip(coords, imgs)]
for triangle, samples in interp_coords.items():
    triangle_images = [located_images[i] for i in triangle]
    located_images.extend(utils.interpolate_image(triangle_images, samples))

# make refocused images


pyplot.figure('triangles')
pyplot.triplot(coords[:, 0], coords[:, 1], regions.simplices)
pyplot.figure('pts')
x, y = coords[0]
pyplot.plot(x, y, 'o')
for x, y in coords[1:]:
    pyplot.plot(x, y, 'x')
pyplot.show()
