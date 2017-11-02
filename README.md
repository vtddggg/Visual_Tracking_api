# Visual_Tracking_api


This is a simple visual tracking interface coding by Python2.7

## Introduction

This repository contains the following contents：

`tutorials` folder contains the implementation and tutorial of various famous tracking algorithms written with ipython-notebook. Learning these notebooks helps you understand the details of the algorithms.

`pyhog` folder includes a implementation of HOG feature. We copied this implementation from https://github.com/dimatura/pyhog

python wrapper script file named `XXXtracker.py`，such as `KCFtracker.py`. These trackers can be integrated into the VOT evaluation process. There is a demo file `vot_demo_tracker.py` representing how to write the wrapper script file.
You can refer to Usage：Evaluate on VOT dataset for getting more information.

Algorithms that have been implemented are as follows:

KCF: High-Speed Tracking with Kernelized Correlation Filters [[PDF]](http://home.isr.uc.pt/~henriques/publications/henriques_tpami2015.pdf)

DSST: Accurate Scale Estimation for Robust Visual Tracking [[PDF]](http://www.cvl.isy.liu.se/en/research/objrec/visualtracking/scalvistrack/ScaleTracking_BMVC14.pdf)

DeepDCF(correlation filter based trackers using deep feature)

## Environment

Python 2.7.12

scikit-image 0.13.0

scipy 0.19.1

matplotlib 1.5.3

numpy 1.13.1

pytorch 0.1.12

opencv 2.4.11

## Usage

### Visualize a sequence

You can run `example.py` to visualize a sequence. Before you run it, note that modify the parameters of `Sequence ()`

```python
sequence = Sequence(path='/YOUR_ROOT_DIR/YOUR_DATASET_NAME', name='THE_NAME_OF_SEQUENCE', region_format='rectangle')

```

For example, visualize a sequence of vot2016:

```python
sequence = Sequence(path='/media/maoxiaofeng/project/GameProject/dataset/vot2016', name='bag',
region_format='rectangle')

Visualize_Tracking(sequence)

```

### Evaluate on VOT dataset

Now the interface is compatible with VOT dataset, use [vot-toolkit](https://github.com/votchallenge/vot-toolkit) to evaluate the tracking algorithm on VOT datasets.

You can download Visual Object Tracking (VOT) challenge datasets through the following links:

[VOT2015](http://data.votchallenge.net/vot2015/vot2015.zip), [VOT2016](http://data.votchallenge.net/vot2016/vot2016.zip), [VOT2014](http://data.votchallenge.net/vot2014/vot2014.zip), [VOT2013](http://data.votchallenge.net/vot2013/vot2013.zip)

Then set up VOT workspace (http://www.votchallenge.net/howto/workspace.html) and integrate trackers into the VOT toolkit (http://www.votchallenge.net/howto/integration.html)

For detail information, please visit VOT official website：http://www.votchallenge.net/

