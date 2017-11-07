import cv2
import os
import shutil
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
from importlib import import_module
import vot
from matplotlib import cm
from numpy import linspace

def Tracking(Sequence, tracker_list, visualize = True):

    if not os.path.exists('results/'):
        os.mkdir("results")

    print 'generate images.txt and region.txt files...'
    with open("images.txt","w") as f:
        while Sequence._frame < len(Sequence._images):
            f.write(Sequence.frame()+'\n')
            Sequence._frame+=1
        Sequence._frame = 0
    with open("region.txt", "w") as f:
        f.write(open(os.path.join(Sequence.seqdir, 'groundtruth.txt'), 'r').readline())

    print 'start tracking...'

    for str in tracker_list:
        print 'tracking using: '+str
        import_module(str)

        if not os.path.exists('results/'+str+'/'+Sequence.name):
            os.makedirs('results/'+str+'/'+Sequence.name)
        shutil.move("output.txt", 'results/'+str+'/'+Sequence.name+'/output.txt')
    os.remove("images.txt")
    os.remove("region.txt")

    print 'Done!!'


    if visualize:
        visulize_result(Sequence, tracker_list)


def visulize_result(Sequence, tracker_list = None, visualize_gt = True):
    fig = plt.figure(1)
    if tracker_list:
        assert (type(tracker_list) == list)
        result = {}
        start = 0.0
        stop = 1.0
        number_of_lines = 1000
        cm_subsection = linspace(start, stop, number_of_lines)

        colors = [cm.jet(x) for x in cm_subsection]
        for str in tracker_list:
            result[str] = open('results/' + str + '/' + Sequence.name+'/output.txt').readlines()
    while Sequence._frame < len(Sequence._images):
        img_rgb = cv2.imread(Sequence.frame())
        plt.clf()
        gt_data = Sequence.groundtruth[Sequence._frame]
        if tracker_list == None:
            pass

        else:
            for str in tracker_list:
                tr_data = vot.convert_region(vot.parse_region(result[str][Sequence._frame]), Sequence._region_format)
                if Sequence._region_format == 'rectangle':

                    tracking_figure_axes = plt.axes()
                    tracking_figure_axes.add_patch(Rectangle(
                        xy=(tr_data.x, tr_data.y),
                        width=tr_data.width,
                        height=tr_data.height,
                        facecolor='none',
                        edgecolor=colors[600/(tracker_list.index(str)+1)],
                    ))

                else:
                    a = []
                    for point in tr_data.points:
                        a.append([point.x, point.y])
                    tr_rect = Polygon(
                        xy=np.array(a),
                        facecolor='none',
                        edgecolor=colors[600/(tracker_list.index(str)+1)],
                    )
                    tracking_figure_axes = plt.axes()
                    tracking_figure_axes.add_patch(tr_rect)


        if visualize_gt:
            if Sequence._region_format == 'rectangle':
                gt_rect = Rectangle(
                xy=(gt_data.x, gt_data.y),
                width=gt_data.width,
                height=gt_data.height,
                facecolor='none',
                edgecolor='r',
            )

                tracking_figure_axes = plt.axes()
                tracking_figure_axes.add_patch(gt_rect)

            else:
                a = []
                for point in gt_data.points:
                    a.append([point.x, point.y])
                gt_rect = Polygon(
                xy=np.array(a),
                facecolor='none',
                edgecolor='r',
            )
                tracking_figure_axes = plt.axes()
                tracking_figure_axes.add_patch(gt_rect)

        plt.imshow(img_rgb)
        plt.draw()
        plt.waitforbuttonpress()
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