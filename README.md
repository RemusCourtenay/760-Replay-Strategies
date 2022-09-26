# 760-Replay-Strategies

## How to use TensorBoard:

Steps:
 - Install the Tensorboard library using pip: `pip install tensorboard` 
 - Use tensorboard to load the logs file, run in your command prompt: `tensorboard --logdir {path-to-logs}`

If you've opened your command prompt in the main project folder then you can use the relative path with the following command:

`tensorboard --logdir logs`

If it's working you should see the following output:

<img src="resources/readme/cmd-output.png" alt="TensorBoard 2.9.1 at http://localhost:6006/">

You can then either click on the link or open your browser and type it into the address bar which should load the tensorboard page:

<img src="resources/readme/tensorboard-main-page.png" alt="TensorBoard main page">

## How to use Scripts

In main.py we are using Script objects to define which strategies we want to test and which models we want to test them on.
Scripts take five inputs as shown below:

<img src="resources/readme/script-setup.png" alt="Script setup">

- The neural network you want to use to run the tests with
- The artist that you want to use to output the results
- The list of strategies that you want to use (these are run separately)
- The datasets that you want to use (also run separately)
- The hyper-parameters to be used

In this example the script will run four different tests with both strategies attempting both datasets. It will also
use the default neural network and artist and run the tests with 1000 memories per task and 10 epochs 
(as defined by the script parameters).

Further explanation of each parameter will be discussed below:

### Neural Network

Currently available options:
- DefaultNeuralNetwork()

The default class is a 6 layer CNN with two convolutional layers, a pooling layer, a flattening layer, and two dense layers. 
It uses the 'adam optimizer', calculates predictions as well as the history, and saves log data to the logs file.
It takes the following optional arguments:
- num_filters_1 = 4 : The number of filters in the initial convolutional layer
- kernel_size_1 = (5, 5) : The shape of the kernel in the first convolutional layer
- input_shape = (28, 28, 1) : The shape of the inputs (generally images)
- max_pooling_shape = (2, 2) : The shape of the MaxPooling2D layer
- num_filters_2 = 8 : The number of filters in the second convolutional layer
- kernel_size_2 = (3, 3) : The shape of the kernel in the second convolutional layer
- activation_type = 'relu' : The activation type for the two convolutional layers and first dense layer
- dense_layer_size = 10 : The size of the last two dense layers

### Artist

Currently available options:
- DefaultArtist()
- PlottingArtist()