# Image Classification CLI

This project provides a command-line interface (CLI) for classifying images using TensorFlow. It leverages a pre-trained model (VGG16) to efficiently categorize images based on the specified parameters.

## Installation

To install the necessary libraries, run the following command:
``` bash
pip install -r requirements.txt
```


- to run
``` bash
python main.py data --num_classes <num_classes> --image_size <image_size> --epochs <epochs>
```

## Parameters
- data: Path to the directory containing the images to classify.
- --num_classes <num_classes>: The number of classes for the classification task.
- --image_size <image_size>: The size to which images will be resized (e.g., 150 for 150x150 pixels).
- --epochs <epochs>: The number of training epochs.

## example
An example of how to run the classification:

``` bash
python main.py data --num_classes 2 --image_size 150 --epochs 5
```
