# SuperStlyeNet: Deep Image Synthesis with Superpixel Based Style Encoder (BMVC 2021)

## Abstract

<details>
  <summary> CLICK ME </summary>
Existing methods for image synthesis utilized a style encoder based on stacks of convolutions and pooling layers to generate style codes from input images. However, the encoded vectors do not necessarily contain local information of the corresponding images since small-scale objects are tended to "wash away" through such downscaling procedures. In this paper, we propose deep image synthesis with superpixel based style encoder, named as SuperStyleNet. First, we directly extract the style codes from the original image based on superpixels to consider local objects. Second, we recover spatial relationships in vectorized style codes based on graphical analysis. Thus, the proposed network achieves high-quality image synthesis by mapping the style codes into semantic labels. Experimental results show that the proposed method outperforms state-of-the-art ones in terms of visual quality and quantitative measurements. Furthermore, we achieve elaborate spatial style editing by adjusting style codes.
</details>

> **SuperStlyeNet: Deep Image Synthesis with Superpixel Based Style Encoder**
> Jonghuyn Kim, Gen Li, Cheolkon Jung, Joongkyu Kim    
> British Machine Vision Conference **BMVC 2021**

[[Paper](https:)]

## Installation

Clone this repo.

Install requirements:
```
  <details>
    <summary> CLICK ME </summary>
  torch==1.2.0
  torchvision==0.4.0
  easydict
  matplotlib
  opencv-python
  glob3
  pillow
  dill
  dominate>=2.3.1 
  scikit-image
  QDarkStyle==2.7
  qdarkgraystyle==1.0.2
  tensorboard==1.14.0
  tensorboardX==1.9
  tqdm==4.32.1
  urllib3==1.25.8
  visdom==0.1.8.9
  </details>
```

## Dataset

This network uses [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ), [Cityscapes](https://www.cityscapes-dataset.com/), and [CMP-Facade](https://cmp.felk.cvut.cz/~tylecr1/facade/) dataset. After downloading this dataset, unzip and save test images in a `./datasets/celeba[dataset name]/train` and `./datasets/celeba[dataset name]/test` folder. 

## Generating images using a pretrained model

After preparing test images, the reconstructed images can be obtained using the pretrained model.

1. Creat a `checkpoint/CelebA` folder. Download pretrained weight from [Google Drive](https://drive.google.com) and upzip this `checkpoint.zip` in the `./checkpoint/celeba` folder.
2. Run `test.py` to generate synthesized images, which will be saved in `./checkpoint/celeba/result`. Save path and details can be edited in `./options/base_options.py` and `./options/test_options.py`.

## Training a new model on personal dataset
We update `train.py` to train SuperStyleNet on personal dataset.

1. Save train and test images in `./datasets/train` and `./datasets/test` folders, respectively.
2. Check your personal setting (i.e., implementation details, save path, and so on) in `./options/base_options.py` and `./options/train_options.py`.
3. Run `train.py` or type 'python train.py' in your terminal.

## Citation
If you use this code for your research, please cite our papers.
