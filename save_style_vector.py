from save_mslic import _style_encoder
import numpy as np
import cv2
import os
from tqdm import tqdm
import glob

save_path = './cityscapes/test/codes'
data_path = './cityscapes/test/image'
anno_path = './cityscapes/test/label'

data = glob.glob(os.path.join(data_path, '*'))

class data_augmentation(object):
    def __init__(self, img_list, anno_path, save_path):

        self.img_list = img_list
        self.anno_path = anno_path
        self.save_path = save_path
        if not os.path.exists(save_path):
            os.mkdir(save_path)

    def open_mask(self, path, width, height, nlabel, isResize=True):

        mask = []
        image_name = path.split('/')[-1]
        anno = cv2.imread(self.anno_path + '/' + image_name)
        if isResize:
            anno = cv2.resize(anno, (width, height), interpolation=cv2.INTER_NEAREST)
        anno = anno[:, :, 0]
        for idx in range(nlabel):
            null = np.zeros_like(anno)
            if idx not in anno:
                mask.append(null)
            else:
                null[anno == idx] = 1
                mask.append(null)
        mask = np.array(mask)

        return mask

    def open_image(self, path, width, height, isResize=True):
        img = cv2.imread(os.path.join(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if isResize:
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
        img = img.astype(np.float32)/255.0
        return img

    def next_batch(self, width, height, nlabel):
        for i in tqdm(self.img_list):

            input_img = i.split('/')[-1]
            number = input_img.split('.')[0]

            mask_img = self.open_mask(i, width, height, nlabel)

            mask_img = np.transpose(np.array(mask_img), (1, 2, 0))

            img = self.open_image(i, width, height)

            styles = _style_encoder(np.array([img]), np.array([mask_img]))

            # num_of_label, batch, 1, 1, 512

            load_styles = np.load(save_path + '/%d.npy' % int(number))
            load_styles[0] = styles[0]
            load_styles = load_styles[:34]

            np.save(save_path + '/%d' % int(number), np.array(load_styles))


data_generator = data_augmentation(data, anno_path, save_path)

data_generator.next_batch(512, 512, nlabel=19) # You should change nlabel depending on dataset
