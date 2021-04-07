# One-Shot Neural Ensemble Architecture Search by Diversity-Guided Search Space Shrinking
This is an official implementation for NEAS presented in CVPR 2021.


## Environment Setup

To set up the enviroment you can easily run the following command:
```buildoutcfg
git clone https://github.com/researchmm/NEAS.git
cd NEAS
conda create -n NEAS python=3.6
conda activate NEAS
sh ./install.sh
# (required) install apex to accelerate the training, a little bit faster than pytorch DistributedDataParallel
cd lib
git clone https://github.com/NVIDIA/apex.git
python ./apex/setup.py install --cpp_ext --cuda_ext
```

#Model Zoo


## Data Preparation 
You need to first download the [ImageNet-2012](http://www.image-net.org/) to the folder `./data/imagenet` and move the validation set to the subfolder `./data/imagenet/val`. To move the validation set, you cloud use the following script: <https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh>

The directory structure is the standard layout as following.
```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class/2
      img4.jpeg
```


## Model Zoo
For evaluation, we provide the checkpoints of our models in [Google Drive](https://drive.google.com/drive/folders/1b3iXPymaCSaXdrI8kuJREvfWY0ycWQXX?usp=sharing).

After downloading the models, you can do the evaluation following the description in *Quick Start - Test*).

Model download links:

Model | FLOPs | Top-1 Acc. % | Top-5 Acc. % | Link 
--- |:---:|:---:|:---:|:---:
NEAS-S | 314M | 77.9 | 93.9 | [Google Drive](https://drive.google.com/file/d/1mZSB45BOp6mui5VerFsOogWky_6h7lsB/view?usp=sharing) 
NEAS-M | 472M | 79.5 | 94.6 | [Google Drive](https://drive.google.com/file/d/1GJCG0nsp8UMUhx4d6ROhHs6diXEvBbd3/view?usp=sharing) 
NEAS-L | 574M | 80.0 | 94.8 | [Google Drive](https://drive.google.com/file/d/1GKCB3-UI3plZSSuE8NPjZ8Q2aygnhiHS/view?usp=sharing)

## Quick Start
We provide *test* code of NEAS as follows.


### Test
To test our trained models, you need to put the downloaded model in `PATH_TO_CKP` (the default path is `./CKP` in root directory.). After that you need to specify the model path in the corresponding config file by changing the `intitial-checkpoint` argument in `./configs/subnets/[SELECTED_MODEL_SIZE].yaml`.

Then, you could use the following command to test the model.
```buildoutcfg
sh ./tools/distribution_test.sh ./configs/subnets/[SELECTED_MODEL_SIZE].yaml
```
The test result will be saved in `./experiments`. You can also add `[--output OUTPUT_PATH]` in `./tools/distribution_test.sh` to specify a path for it as well.

## To Do List

- [x] Test code
- [ ] Retrain code
- [ ] Search code


## BibTex
```
@article{NEAS,
  title={One-Shot Neural Ensemble Architecture Search by Diversity-Guided Search Space Shrinking},
  author={Chen, Minghao and Peng, Houwen and Fu, Jianlong and Ling, Haibin},
  journal={arXiv preprint arXiv:2104.00597},
  year={2021}
}
```

