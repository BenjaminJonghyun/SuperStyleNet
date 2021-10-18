# SuperStyleNet: Deep Image Synthesis with Superpixel Based Style Encoder (BMVC 2021)

![Mix_comp](https://user-images.githubusercontent.com/42399549/137694588-28f522ee-e9aa-480c-8f85-eba8f1ebe0e6.png)
**Figure:** Style mixing with multiple style images. The style vectors are replaced from source to style image on given semantic masks.

![SPSE](https://user-images.githubusercontent.com/42399549/137692560-ccb7e96e-6b9a-417c-8bbe-97db01205ea2.png)
**Figure:** Superpixel based Style Encoding. To extract style codes of a specific semantic mask, we convert the input image into the five-dimensional space and cluster it in the semantic mask into superpixels. Thereafter, pixel values in each superpixel are averaged to obtain a style code.

## Abstract

<details>
  <summary> CLICK ME </summary>
Existing methods for image synthesis utilized a style encoder based on stacks of convolutions and pooling layers to generate style codes from input images. However, the encoded vectors do not necessarily contain local information of the corresponding images since small-scale objects are tended to "wash away" through such downscaling procedures. In this paper, we propose deep image synthesis with superpixel based style encoder, named as SuperStyleNet. First, we directly extract the style codes from the original image based on superpixels to consider local objects. Second, we recover spatial relationships in vectorized style codes based on graphical analysis. Thus, the proposed network achieves high-quality image synthesis by mapping the style codes into semantic labels. Experimental results show that the proposed method outperforms state-of-the-art ones in terms of visual quality and quantitative measurements. Furthermore, we achieve elaborate spatial style editing by adjusting style codes.
</details>

> **SuperStlyeNet: Deep Image Synthesis with Superpixel Based Style Encoder**
> 
> Jonghuyn Kim, Gen Li, Cheolkon Jung, Joongkyu Kim    
> British Machine Vision Conference **BMVC 2021**

[[Paper](https:)]

## Installation

Clone this repo.

Install requirements:

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

## Dataset

1. This network uses [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ), [Cityscapes](https://www.cityscapes-dataset.com/), and [CMP-Facade](https://cmp.felk.cvut.cz/~tylecr1/facade/) datasets. After downloading these datasets, unzip and save train and test images as follows: 
```
dataset
  ├── celeba
  |    ├── train
  |    |     ├── images
  |    |     ├── labels
  |    |     └── codes
  |    ├── test
  |    |     ├── images
  |    |     ├── labels
  |    |     └── codes
  ├── cityscapes
  |    ├── train
  |    |     ├── images
  |    |     ├── labels
  |    |     └── codes
  |    ├── test
  |    |     ├── images
  |    |     ├── labels
  |    |     └── codes          
```
2. Download style codes in each dataset from [Google Drive](https://drive.google.com). After downloading them, unzip and save in `./dataset/[dataset name]/[train or test]/codes`. **To extract style codes using SPSE, it requires a lot of time. Thereby, we provide all style codes of three datasets.**

## Generating images using a pretrained model with style codes

After preparing test images, the reconstructed images can be obtained using the pretrained model.

1. Creat a `checkpoint/celeba` folder. Download pretrained weight from [Google Drive](https://drive.google.com) and upzip this `checkpoint.zip` in the `./checkpoint/celeba` folder.
2. Run `test.py` to generate synthesized images with a below code, which will be saved in `./checkpoint/celeba/result`. Save path and details can be edited in `./options/base_options.py` and `./options/test_options.py`.
```
python test.py --name celeba --load_size 256 --crop_size 256 --dataset_mode custom --label_dir datasets/celeba/test/labels --image_dir datasets/celeba/test/images --label_nc 19 --instance_dir datasets/celeba/test/codes --which_epoch 50 --gpu_ids 0
```

## Training a new model on personal dataset

### For CelebAMask-HQ
1. Check your personal setting (i.e., implementation details, save path, and so on) in `./options/base_options.py` and `./options/train_options.py`.
2. Run `train.py`.
```
python train.py --name celeba --gpu_ids 0,1,2,3 --batchSize 32 --load_size 256 --crop_size 256 --dataset_mode custom --label_nc 19 --label_dir datasets/celeba/train/labels --image_dir datasets/celeba/train/images --instance_dir datasets/celeba/train/codes
```

### For personal dataset
1. Save train and test images with labels in `./datasets/[dataset name]/train/[images or labels]` and `./datasets/[dataset name]/test/[images or labels]` folders, respectively.
2. Run `save_style_vector.py` to extract and save style vectors. This process requires a lot of time.
3. Check your personal setting (i.e., implementation details, save path, and so on) in `./options/base_options.py` and `./options/train_options.py`.
4. Run `train.py`.
```
python train.py --name personal_data --gpu_ids 0,1,2,3 --batchSize 32 --load_size 256 --crop_size 256 --dataset_mode custom --label_nc 19 --label_dir datasets/[dataset name]/train/labels --image_dir datasets/[dataset name]/train/images --instance_dir datasets/[dataset name]/train/codes
```

## Citation
If you use this code for your research, please cite our papers.
