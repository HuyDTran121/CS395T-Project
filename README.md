## Special Notes
To switch attention mechanisms, change the BLOCK_TYPE in model.py line 289
The code this project was based can be found at [link](https://github.com/odegeasslbc/FastGAN-pytorch)

## 0. Data
The datasets used in the paper can be found at [link](https://drive.google.com/drive/folders/1GqSaMMkFn4-POR34e8PbZJfgfJXMavku?usp=sharing).

 

## 1. Description
The code is structured as follows:
* models.py: all the models' structure definition.

* operation.py: the helper functions and data loading methods during training.

* train.py: the main entry of the code, execute this file to train the model, the intermediate results and checkpoints will be automatically saved periodically into a folder "train_results".

* eval.py: generates images from a trained generator into a folder, which can be used to calculate FID score.

* benchmarking: the functions we used to compute FID are located here, it automatically downloads the pytorch official inception model. 

* lpips: this folder contains the code to compute the LPIPS score, the inception model is also automatically download from official location.

* scripts: this folder contains many scripts you can use to play around the trained model. Including: 
    1. style_mix.py: style-mixing as introduced in the paper;
    2. generate_video.py: generating a continuous video from the interpolation of generated images;
    3. find_nearest_neighbor.py: given a generated image, find the closest real-image from the training set;
    4. train_backtracking_one.py: given a real-image, find the latent vector of this image from a trained Generator.

## 2. How to run
Place all your training images in a folder, and simply call
```
python train.py --path /path/to/RGB-image-folder
```
You can also see all the training options by:
```
python train.py --help
```
The code will automatically create a new folder (you have to specify the name of the folder using --name option) to store the trained checkpoints and intermediate synthesis results.

Once finish training, you can generate 100 images (or as many as you want) by:
```
cd ./train_results/name_of_your_training/
python eval.py --n_sample 100 
```

