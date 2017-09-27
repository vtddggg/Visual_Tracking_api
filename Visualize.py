import matplotlib.pyplot as plt
from skimage import io
from matplotlib.patches import Rectangle, Polygon
import numpy as np

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