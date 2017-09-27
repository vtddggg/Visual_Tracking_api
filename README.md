# Visual_Tracking_api


This is a simple visual tracking interface coding by Python2.7

Now the interface is compatible with VOT dataset

You can download Visual Object Tracking (VOT) challenge datasets through the following links:

[VOT2015](http://data.votchallenge.net/vot2015/vot2015.zip), [VOT2016](http://data.votchallenge.net/vot2016/vot2016.zip)

Then unzip the dataset into a your directory

## Environment

Python 2.7.12

scikit-image 0.13.0

opencv 2.4.11

matplotlib 1.5.3

numpy 1.13.1

## Usage

You can run `example.py` to understand how to use. Before you run it, note that modify the parameters of `Sequence ()`

```python
sequence = Sequence(path='/YOUR_ROOT_DIR/vot2016', name='THE_NAME_OF_SEQUENCE', region_format='rectangle')

```

