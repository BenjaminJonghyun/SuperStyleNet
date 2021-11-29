"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

# python test.py --name slic_gsas_test --load_size 256 --crop_size 256 --dataset_mode custom --label_dir datasets/CelebA-HQ/test/labels --image_dir datasets/CelebA-HQ/test/images --label_nc 19 --instance_dir datasets/CelebA-HQ/test/codes_fb --which_epoch 50 --gpu_ids 3
# python train.py --name slic_gsas_facade_test --gpu_ids 0,1,2,3 --batchSize 32 --load_size 256 --crop_size 256 --dataset_mode custom --label_nc 12 --label_dir datasets/Facade/train/labels --image_dir datasets/Facade/train/images --instance_dir datasets/Facade/train/codes
#  python test.py --name slic_gsas_city_test --gpu_ids 3 --load_size 256 --crop_size 256 --dataset_mode custom --label_nc 34 --label_dir datasets/city/test/labels --image_dir datasets/city/test/images --instance_dir datasets/city/test/codes --which_epoch 126

import os
from collections import OrderedDict

import data
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from util.visualizer import Visualizer
from util import html
from tqdm import tqdm

opt = TestOptions().parse()
opt.status = 'test'

dataloader = data.create_dataloader(opt)

model = Pix2PixModel(opt)
model.eval()

visualizer = Visualizer(opt)

# create a webpage that summarizes the all results
web_dir = os.path.join(opt.results_dir, opt.name,
                       '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir,
                    'Experiment = %s, Phase = %s, Epoch = %s' %
                    (opt.name, opt.phase, opt.which_epoch))

# test
ssim = 0
psnr = 0
nrmse = 0
for i, data_i in enumerate(tqdm(dataloader)):
    if i * opt.batchSize >= opt.how_many:
        break

    generated = model(data_i, mode='inference')

    img_path = data_i['path']
    for b in range(generated.shape[0]):
        #print('process image... %s' % img_path[b])
        visuals = OrderedDict([('input_label', data_i['label'][b]),
                               ('synthesized_image', generated[b]),
                               ('real_image', data_i['image'][b])])
        visualizer.save_images(webpage, visuals, img_path[b:b + 1])
        #_ssim, _psnr, _nrmse = visualizer.cal_eval(visuals)
        #ssim += _ssim/500
        #psnr += _psnr/500
        #nrmse += _nrmse/500
#print(ssim, psnr, nrmse)
webpage.save()


