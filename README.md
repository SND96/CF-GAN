#  Cognitive Fashion using Generative Adversarial Networks

## CycleGAN

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

---

## LRGAN

#### Prerequisites
1. Python 2.7 with Pytorch

#### Input format
* Add the dataset that you want to use to the 'datasets' folder according to this format : Data/Folders_of_images/image.jpg
* Note that for the pytorch dataloader to work, the folder that you pass must contain sub-folders containing the images. Pytorch assumes each of these sub-folders is a class. 
* Since classes are not important for the training of this model, there can be a separate folder for each image

#### Running the script
* Use this command to run the script with the appropriate arguments. More information about the arguments is found in the train.py file
```
python train.py       --dataset street       --dataroot datasets/street       --ntimestep 2       --imageSize 64       --maxobjscale 1.2       --niter 100       --session 1
```

#### Samples
* The sample images will be saved in the images folder by default. Modify the location with the appropriate argument in the script call.
