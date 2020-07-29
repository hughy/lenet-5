# lenet-5
The LeNet-5 architecture was an early convolutional neural network. Yann LeCun et al. developed the network architecture in a 1998 paper and applied it to classify handwritten digits.

http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf

## Network Architecture
The LeNet-5 network consists of seven layers, shown below:

![LeNet-5 Architecture](docs/lenet-5.png)

## Implementation Details
The second convolutional layer, C2, in LeCun's implementation features incomplete connections from the previous layer. This incomplete connection forces the network to break symmetry and reduces the number of parameters to train. This implementation forgoes this detail in order to simplify the implementation of the network architecture.

In order to feed the output of the C5 layer to the following fully connected layer, F6, we add an additional operation to flatten the output from a 120x1x1 volume to a 1x120 matrix.
## Usage
### Setup
This project uses `poetry` to manage Python dependencies and virtual environments. You can use `pip` to install `poetry`:

    pip install poetry

Use `poetry` to install project dependencies with

    poetry install

### Training
To train the LeNet-5 model on the MNIST dataset run the command below:

    poetry run python -m lenet_5.train