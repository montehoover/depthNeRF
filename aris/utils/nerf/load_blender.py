import os
import torch
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2


trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


def load_blender_data(basedir, half_res=False, testskip=1, use_depths=False):
    # If we want depth images then 'train' must be first so we can grab a placeholder depth image
    # We only use depth images for training, but we read in dummy images for val and test
    # just to keep the indexing consistent with RGB images.
    # Note, these are technically disparity images, so equal to 1/depth.
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]

    if use_depths:
        all_depth_imgs = []
        get_first = True
        placeholder = ""

    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        depth_imgs = []
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip

        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))

            if use_depths:
                if s == 'train':
                    depth_fname = os.path.join(basedir, frame['file_path'] + "_depth_0001" +'.png')
                    assert os.path.exists(fname), "The config said to use depth but no depth images found: {}".format(fname)
                    if get_first:
                        placeholder = depth_fname
                        get_first = False
                # Load a dummy depth image for val and test
                else:
                    fname = placeholder
                depth_imgs.append(imageio.imread(depth_fname))

        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        if use_depths:
            depth_imgs = (np.array(depth_imgs) / 255.).astype(np.float32) # (n_imgs, H, W, 4)
            # All 3 RGB channels are identical and we don't need alpha. Keep the first channel.
            depth_imgs = depth_imgs[:,:,:,0]  # (n_imgs, H, W)

        all_imgs.append(imgs)
        all_poses.append(poses)
        if use_depths:
            all_depth_imgs.append(depth_imgs)

    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]

    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    if use_depths:
        depth_imgs = np.concatenate(all_depth_imgs, 0)

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)

    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        if use_depths:
            depth_imgs_half_res = np.zeros((depth_imgs.shape[0], H, W))
            for i, depth_img in enumerate(depth_imgs):
                depth_imgs_half_res[i] = cv2.resize(depth_img, (W, H), interpolation=cv2.INTER_AREA)
            depth_imgs = depth_imgs_half_res

    if not use_depths:
        depth_imgs = None
    else:
        depth_imgs = np.expand_dims(depth_imgs, axis=-1)  # (n_imgs, H, W, 1)

    return imgs, poses, render_poses, [H, W, focal], i_split, depth_imgs
