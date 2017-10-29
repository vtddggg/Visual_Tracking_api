# Visual_Tracking_api


This is a simple visual tracking interface coding by Python2.7

## VOT

Now the interface is compatible with VOT dataset, you can use [vot-toolkit](https://github.com/votchallenge/vot-toolkit) to evaluate the tracking algorithm on VOT datasets.

You can download Visual Object Tracking (VOT) challenge datasets through the following links:

[VOT2015](http://data.votchallenge.net/vot2015/vot2015.zip), [VOT2016](http://data.votchallenge.net/vot2016/vot2016.zip)

For detail information, please visit VOT official website：http://www.votchallenge.net/

## Environment

Python 2.7.12

scikit-image 0.13.0

scipy 0.19.1

matplotlib 1.5.3

numpy 1.13.1

pytorch 0.1.12

## Introduction

### Visualize

You can run `example.py` to visualize a sequence. Before you run it, note that modify the parameters of `Sequence ()`

```python
sequence = Sequence(path='/YOUR_ROOT_DIR/vot2016', name='THE_NAME_OF_SEQUENCE', region_format='rectangle')

```

### Tutorials

`tutorials` folder contains the implementation and tutorial of various famous tracking algorithms written with ipython-notebook. Learning these notebooks helps you understand the details of the algorithms.

