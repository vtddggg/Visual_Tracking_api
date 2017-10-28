import matplotlib.pyplot as plt
from skimage import io
from matplotlib.patches import Rectangle, Polygon
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid

def Visualize_Tracking(Sequence, visualize_type = 'groundtruth'):

    while Sequence._frame < len(Sequence._images):
        img_rgb = io.imread(Sequence.frame())
        fig = plt.figure(1)
        plt.clf()

        if visualize_type == 'groundtruth':

            data = Sequence.groundtruth[Sequence._frame]

        else:

            if len(Sequence._result) ==0:
                raise Exception('No result, error!!')

            data = Sequence._result[Sequence._frame]

        if Sequence._region_format == 'rectangle':
            tracking_rect = Rectangle(
            xy=(data.x, data.y),
            width=data.width,
            height=data.height,
            facecolor='none',
            edgecolor='r',
            )

            tracking_figure_axes = plt.axes()
            tracking_figure_axes.add_patch(tracking_rect)

        else:
            a = []
            for point in data.points:
                a.append([point.x, point.y])
            tracking_rect = Polygon(
            xy=np.array(a),
            facecolor='none',
            edgecolor='r',
            )
            tracking_figure_axes = plt.axes()
            tracking_figure_axes.add_patch(tracking_rect)

        plt.imshow(img_rgb)
        plt.colorbar()
        plt.draw()
        plt.waitforbuttonpress(0.001)
        Sequence._frame += 1

def imshow_grid(images, shape=[3, 10]):
    """Plot images in a grid of a given shape."""
    fig = plt.figure(1)
    grid = ImageGrid(fig, 111, nrows_ncols=shape, axes_pad=0.05)

    size = shape[0] * shape[1]
    for i in range(size):
        grid[i].axis('off')
        grid[i].imshow(images[i]/np.max(images[i]),cmap=plt.cm.gray)  # The AxesGrid object work as a list of axes.

    plt.show()