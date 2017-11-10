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
                        edgecolor=colors[tracker_list.index(str)*number_of_lines / len(tracker_list)],
                    ))
                    tracking_figure_axes.text(100, 20*(tracker_list.index(str)+1), str,
                            verticalalignment='bottom', horizontalalignment='right',
                            color=colors[tracker_list.index(str)*number_of_lines / len(tracker_list)], fontsize=15)

                else:
                    a = []
                    for point in tr_data.points:
                        a.append([point.x, point.y])
                    tr_rect = Polygon(
                        xy=np.array(a),
                        facecolor='none',
                        edgecolor=colors[tracker_list.index(str)*number_of_lines / len(tracker_list)],
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

def precision_plot(Sequence, tracker_list):
    start = 0.0
    stop = 1.0
    number_of_lines = 1000
    cm_subsection = linspace(start, stop, number_of_lines)

    colors = [cm.jet(x) for x in cm_subsection]
    fig = plt.Figure()
    plt.xlabel('Threshold')
    plt.ylabel('Precision')
    plt.ylim(0, 1)
    max_threshold = 50
    gt = [[data.y + data.height / 2, data.x + data.width / 2] for data in Sequence.groundtruth]
    gt = np.array(gt)
    for str in tracker_list:
        precisions = np.zeros(shape = [max_threshold])
        result = np.loadtxt('results/' + str + '/' + Sequence.name + '/output.txt',delimiter=',')
        positions = result[:,[1,0]]+result[:,[3,2]]/2
        distance = np.sqrt(np.sum(np.power(positions-gt,2),1))
        for p in range(max_threshold):
            precisions[p] = float(np.count_nonzero(distance<(p+1)))/distance.shape[0]
        plt.plot(precisions,color =colors[tracker_list.index(str)*number_of_lines / len(tracker_list)], label =str)
    plt.legend()
    plt.show()

def overlap_plot(Sequence, tracker_list):
    start = 0.0
    stop = 1.0
    number_of_lines = 1000
    cm_subsection = linspace(start, stop, number_of_lines)

    colors = [cm.jet(x) for x in cm_subsection]
    fig = plt.Figure()
    plt.xlabel('Threshold')
    plt.ylabel('Overlap')
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    inter_p = 100
    gt = [[data.x, data.y, data.width, data.height] for data in Sequence.groundtruth]
    gt = np.array(gt)
    for str in tracker_list:
        Thresholds = np.arange(0,1,1.0/inter_p)+1.0/inter_p
        overlap_precision = np.zeros(shape=[inter_p])

        result = np.loadtxt('results/' + str + '/' + Sequence.name + '/output.txt', delimiter=',')
        endX = np.max(np.vstack((result[:,0]+result[:,2],gt[:,0]+gt[:,2])),axis=0)
        startX = np.min(np.vstack((result[:,0], gt[:,0])),axis=0)
        width = result[:,2]+gt[:,2]-(endX-startX)
        width[width < 0] = 0

        endY = np.max(np.vstack((result[:, 1] + result[:, 3], gt[:, 1] + gt[:, 3])), axis=0)
        startY = np.min(np.vstack((result[:, 1], gt[:, 1])), axis=0)
        height = result[:, 3] + gt[:, 3] - (endY - startY)
        height[height < 0] = 0

        Area = np.multiply(width,height)
        Area1 = np.multiply(result[:,2],result[:,3])
        Area2 = np.multiply(gt[:,2],gt[:,3])
        overlap_ratio = np.divide(Area,Area1+Area2-Area)

        for p in range(inter_p):
            overlap_precision[p] = float(np.count_nonzero(overlap_ratio > Thresholds[p])) / overlap_ratio.shape[0]
        plt.plot(overlap_precision, Thresholds, color=colors[tracker_list.index(str)*number_of_lines / len(tracker_list)], label=str)
    plt.legend()
    plt.show()







def imshow_grid(images, shape=[3, 10]):
    """Plot images in a grid of a given shape."""
    fig = plt.figure(1)
    grid = ImageGrid(fig, 111, nrows_ncols=shape, axes_pad=0.05)

    size = shape[0] * shape[1]
    for i in range(size):
        grid[i].axis('off')
        grid[i].imshow(images[i]/np.max(images[i]),cmap=plt.cm.gray)  # The AxesGrid object work as a list of axes.

    plt.show()