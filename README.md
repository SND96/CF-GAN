#  Cognitive Fashion using Generative Adversarial Networks

### CycleGAN

#### Prerequisites
1. Python 3 with Keras 2.0 and Tensorflow (1.7) backend
2. Jupyter Notebook

#### Input format
* There will be two datasets which will used for training.
* Divide both the datasets into a training set and test set and label them as TrainA and TestA, TrainB and TestB.
* Add the directory containing these folders to the path used to load the dataset in the notebook.
* Adjust the checkpointing for saving the models to your suiting. Currently it saves two models, one being made every 10 epochs
Also add the path to where you want the model to be saved.

#### Training
* For adversarial loss use CycleGAN-keras-ECI.ipynb
* For Wasserstein loss use CycleGAN-keras-wgan.ipynb

#### Testing
* Use CycleGAN-keras-test.ipynb and add the paths to where the model has been saved.

