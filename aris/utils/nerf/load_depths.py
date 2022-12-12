import os
import torch
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2


def load_depth_data(basedir, half_res=False, testskip=1):
    # Note this requires 'train' to be first so we can grab a placeholder depth image
    # We only use depth images for training, but we read in dummy images for val and test
    # just to keep the indexing consistent with RGB images.
    # Note, these are technically disparity images, so equal to 1/depth.
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    get_first = True
    placeholder = ""
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip

        for frame in meta['frames'][::skip]:
            if s == 'train':
                fname = os.path.join(basedir, frame['file_path'] + "_depth_0001" +'.png')
                assert os.path.exists(fname), "Depth image not found: {}".format(fname)
                if get_first:
                    placeholder = fname
                    get_first = False
            else:
                fname = placeholder
            imgs.append(imageio.imread(fname))
        imgs = (np.array(imgs) / 255.)[:, :, 0].astype(np.float32) # All 4 channels are identical. Keep the first one.
        imgs = imgs[:, :, None] # (H x W x 1)
        all_imgs.append(imgs)

    imgs = np.concatenate(all_imgs, 0)
    return imgs
