import logging

# import cv2
import imageio
import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from torch import Tensor
import torch.nn.functional as F
from tqdm import tqdm, trange

import os, sys
import time

import matplotlib.pyplot as plt

from aris.core.io import build_integrator
from aris.integrator import Integrator
# from aris.utils.image_utils import save_image, tonemap_image
# from aris.utils.render_utils import get_coords_multisample

from aris.utils.nerf.run_helpers import get_rays                    # general
from aris.utils.nerf.run_helpers import img2mse, mse2psnr, to8b     # metrics
from aris.utils.nerf.run_helpers import pose_spherical              # spherical coords to rot

from aris.utils.nerf.load_llff import load_llff_data
from aris.utils.nerf.load_deepvoxels import load_dv_data
from aris.utils.nerf.load_blender import load_blender_data
from aris.utils.nerf.load_LINEMOD import load_LINEMOD_data
from aris.utils.nerf.load_depths import load_depth_data


logger = logging.getLogger(__name__)
window_name = "main"
np.random.seed(0)



@hydra.main(version_base=None, config_path="config", config_name="nerf")
def main(cfg: HydraConfig = None):
    if cfg.device == 'cuda':
        assert torch.cuda.is_available(), "CUDA GPU is not available (try device=cpu)"
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # save everything to out_dir
    # out_dir = Path(HydraConfig().get().run.dir)
    out_dir = cfg.basedir

    # print and save config for debugging
    config_yaml = OmegaConf.to_yaml(cfg)
    logger.info(config_yaml)
    with open(out_dir + "/config.yaml", mode="w", encoding="utf-8") as f:
        f.write(config_yaml + "\n")

    # Load data
    K = None
    if cfg.nerf.dataset.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(cfg.nerf.dataset.datadir, cfg.nerf.dataset.llff.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=cfg.nerf.dataset.llff.spherify)
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        logger.info(f'Loaded llff {images.shape} {render_poses.shape} {hwf} {cfg.nerf.dataset.datadir}')
        if not isinstance(i_test, list):
            i_test = [i_test]

        if cfg.nerf.dataset.llff.spherify > 0:
            logger.info(f'Auto LLFF holdout, {cfg.nerf.dataset.llff.spherify}')
            i_test = np.arange(images.shape[0])[::cfg.nerf.dataset.llff.spherify]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

        logger.info('DEFINING BOUNDS')
        if cfg.nerf.dataset.llff.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.

        else:
            near = 0.
            far = 1.
        logger.info(f'NEAR FAR {near} {far}')

    elif cfg.nerf.dataset.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split, depth_images = load_blender_data(cfg.nerf.dataset.datadir, cfg.nerf.dataset.blendr.half_res, cfg.nerf.dataset.testskip, cfg.nerf.training.use_depths)
        if cfg.nerf.training.use_depths:
            assert depth_images is not None, "The config said to use depth but depth images were not successfully loaded."
        logger.info(f'Loaded blender {images.shape} {render_poses.shape} {hwf} {cfg.nerf.dataset.datadir}')
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if cfg.nerf.dataset.blendr.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif cfg.nerf.dataset.dataset_type == 'LINEMOD':
        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(cfg.nerf.dataset.datadir, cfg.nerf.dataset.blendr.cfg.nerf.dataset.blendr.half_res, cfg.dataset.testskip)
        logger.info(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
        logger.info(f'[CHECK HERE] near: {near}, far: {far}.')
        i_train, i_val, i_test = i_split

        if cfg.nerf.dataset.blendr.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif cfg.nerf.dataset.dataset_type == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = load_dv_data(scene=cfg.nerf.dataset.deepvoxels.shape,
                                                                 basedir=cfg.nerf.dataset.datadir,
                                                                 testskip=cfg.nerf.dataset.testskip)

        logger.info(f'Loaded deepvoxels {images.shape} {render_poses.shape} {hwf} {cfg.nerf.dataset.datadir}')
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.

    else:
        logger.error(f'Unknown dataset type {cfg.nerf.dataset.dataset_type}. Exiting.')
        return

    # if cfg.nerf.training.use_depths:
    #     depth_images = load_depth_data(cfg.nerf.dataset.datadir)

    # configure the integrator
    integrator = build_integrator(cfg.integrator)

    # cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    if cfg.nerf.rendering.render_test:
        render_poses = np.array(poses[i_test])

    result = torch.zeros(H, W, 3)


    # Create log dir and copy the config file
    basedir = cfg.basedir
    expname = cfg.nerf.experiment
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'cfg.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(cfg)):
            attr = getattr(cfg, arg)
            file.write('{} = {}\n'.format(arg, attr))

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = integrator.create_nerf(cfg)
    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(cfg.device)

    # Short circuit if only rendering out from trained model
    if cfg.nerf.rendering.render_only:
        logger.info('RENDER ONLY')
        with torch.no_grad():

            # Render video
            if cfg.nerf.rendering.render_video:
                # render a full video using pre-defined poses
                if cfg.gui == True:
                    logger.error("Error: render_video and gui cannot be true at the same time. Continuing without gui.")

                if cfg.nerf.rendering.render_test:
                    # render_test switches to test poses
                    images = images[i_test]
                else:
                    # default is smoother render_poses path
                    images = None

                testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if cfg.nerf.rendering.render_test else 'path', start))
                os.makedirs(testsavedir, exist_ok=True)
                logger.info(f'test poses shape {render_poses.shape}')

                rgbs, _ = integrator.render_path(render_poses, hwf, K, cfg.nerf.training.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=cfg.nerf.rendering.render_factor)
                logger.info(f'Done rendering {testsavedir}')
                imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

            # Render single image
            else:
                # print(render_poses[10,:,:])

                # TODO: determine appropriate poses for other datasets

                theta = cfg.nerf.rendering.pose.theta
                phi = cfg.nerf.rendering.pose.phi
                r = cfg.nerf.rendering.pose.radius
                logger.info(f'rendering pose (theta={theta}, phi={phi}, r={r})')

                render_pose = pose_spherical(theta, phi, r).view(1,4,4)
                testsavedir = os.path.join(basedir, expname)
                os.makedirs(testsavedir, exist_ok=True)
                logger.info(f'pose shape {render_pose.shape}')

                rgbs, _ = integrator.render_path(render_pose, hwf, K, cfg.nerf.training.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=cfg.nerf.rendering.render_factor)
                logger.info(f'Done rendering {testsavedir}')

                # TODO: add live gui?

            integrator.on_render_ended()

            return

    # Prepare raybatch tensor if batching random rays
    N_rand = cfg.nerf.training.N_rand
    use_batching = not cfg.nerf.training.no_batching
    if use_batching:
        # For random ray batching
        logger.info('get rays')
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        logger.info('done, concats')
        rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
        rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = rays_rgb.astype(np.float32)
        logger.info('shuffle rays')
        np.random.shuffle(rays_rgb)

        if cfg.nerf.training.use_depths:
            rays_depth = np.concatenate([rays, depth_images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
            rays_depth = np.transpose(rays_depth, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
            rays_depth = np.stack([rays_depth[i] for i in i_train], 0) # train depth images only
            rays_depth = np.reshape(rays_depth, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
            rays_depth = rays_depth.astype(np.float32)
            np.random.shuffle(rays_depth)

        logger.info('done')
        i_batch = 0

    # Move training data to GPU
    if use_batching:
        images = torch.Tensor(images).to(cfg.device)
    poses = torch.Tensor(poses).to(cfg.device)
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(cfg.device)
    if cfg.nerf.training.use_depths:
        depth_images = torch.Tensor(depth_images).to(cfg.device)
        if use_batching:
            rays_depth = torch.Tensor(rays_depth).to(cfg.device)

    N_iters = cfg.nerf.training.epochs  # originally was 200000 + 1
    logger.info(f'Begin')
    logger.info(f'TRAIN views are {i_train}')
    logger.info(f'TEST views are {i_test}')
    logger.info(f'VAL views are {i_val}')

    # Summary writers
    # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))

    start = start + 1
    for i in trange(start, N_iters):
        time0 = time.time()

        # Sample random ray batch
        if use_batching:

            # Random over all images
            batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]
            if cfg.nerf.training.use_depths:
                batch_depth = rays_depth[i_batch:i_batch+N_rand]
                batch_depth = torch.transpose(batch_depth, 0, 1)
                batch_rays_depth, target_depth_s = batch_depth[:2], batch_depth[2]

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                logger.info("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0

        else:
            # Random from one image
            img_i = np.random.choice(i_train)
            target = images[img_i]
            target = torch.Tensor(target).to(cfg.device)
            pose = poses[img_i, :3,:4]
            if cfg.nerf.training.use_depths:
                target_depth = depth_images[img_i]
                target_depth = torch.Tensor(target_depth).to(cfg.device)

            if N_rand is not None:
                rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

                if i < cfg.nerf.training.precrop_iters:
                    dH = int(H//2 * cfg.nerf.training.precrop_frac)
                    dW = int(W//2 * cfg.nerf.training.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH),
                            torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                        ), -1)
                    if i == start:
                        logger.info(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {cfg.nerf.training.precrop_iters}")
                else:
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

                coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                if cfg.nerf.training.use_depths:
                    target_depth_s = target_depth[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 1)

        #####  Core optimization loop  #####
        rgb, disp, acc, extras = integrator.render(H, W, K, chunk=cfg.nerf.training.chunk, rays=batch_rays,
                                                   verbose=i < 10, retraw=True,
                                                   **render_kwargs_train)

        optimizer.zero_grad()
        img_loss = img2mse(rgb, target_s)


        trans = extras['raw'][...,-1]
        if cfg.nerf.training.use_depths:
            disp = disp[..., None]  # (N_rand, 1)
            depth_loss = torch.mean((disp - target_depth_s) ** 2)
            loss = img_loss + cfg.nerf.training.depth_lambda * depth_loss
        else:
            loss = img_loss
        psnr = mse2psnr(img_loss)

        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        loss.backward()
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = cfg.nerf.training.lrate_decay * 1000
        new_lrate = cfg.nerf.training.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        dt = time.time()-time0
        # logger.info(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        # Rest is logging
        if i%cfg.nerf.logging.i_weights==0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            logger.info(f'Saved checkpoints at {path}')

        if i%cfg.nerf.logging.i_video==0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps = integrator.render_path(render_poses, hwf, K, cfg.nerf.training.chunk, render_kwargs_test)
            logger.info(f'Done, saving {rgbs.shape} {disps.shape}')
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps), fps=30, quality=8)

            if cfg.nerf.rendering.use_viewdirs:
                render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
                with torch.no_grad():
                    rgbs_still, _ = integrator.render_path(render_poses, hwf, K, cfg.nerf.training.chunk, render_kwargs_test)
                render_kwargs_test['c2w_staticcam'] = None
                imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)

        if i%cfg.nerf.logging.i_testset==0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            logger.info(f'test poses shape {poses[i_test].shape}')
            with torch.no_grad():
                integrator.render_path(torch.Tensor(poses[i_test]).to(cfg.device), hwf, K, cfg.nerf.training.chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)
            logger.info('Saved test set')



        if i%cfg.nerf.logging.i_print==0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")
        """
            print(expname, i, psnr.numpy(), loss.numpy(), global_step.numpy())
            print('iter time {:.05f}'.format(dt))

            with tf.contrib.summary.record_summaries_every_n_global_steps(cfg.nerf.logging.i_print):
                tf.contrib.summary.scalar('loss', loss)
                tf.contrib.summary.scalar('psnr', psnr)
                tf.contrib.summary.histogram('tran', trans)
                if cfg.nerf.rendering.N_importance > 0:
                    tf.contrib.summary.scalar('psnr0', psnr0)


            if i%cfg.nerf.logging.i_img==0:

                # Log a rendered validation view to Tensorboard
                img_i=np.random.choice(i_val)
                target = images[img_i]
                pose = poses[img_i, :3,:4]
                with torch.no_grad():
                    rgb, disp, acc, extras = integrator.render(H, W, focal, chunk=cfg.nerf.training.chunk, c2w=pose,
                                                        **render_kwargs_test)

                psnr = mse2psnr(img2mse(rgb, target))

                with tf.contrib.summary.record_summaries_every_n_global_steps(cfg.nerf.logging.i_img):

                    tf.contrib.summary.image('rgb', to8b(rgb)[tf.newaxis])
                    tf.contrib.summary.image('disp', disp[tf.newaxis,...,tf.newaxis])
                    tf.contrib.summary.image('acc', acc[tf.newaxis,...,tf.newaxis])

                    tf.contrib.summary.scalar('psnr_holdout', psnr)
                    tf.contrib.summary.image('rgb_holdout', target[tf.newaxis])


                if cfg.nerf.rendering.N_importance > 0:

                    with tf.contrib.summary.record_summaries_every_n_global_steps(cfg.nerf.logging.i_img):
                        tf.contrib.summary.image('rgb0', to8b(extras['rgb0'])[tf.newaxis])
                        tf.contrib.summary.image('disp0', extras['disp0'][tf.newaxis,...,tf.newaxis])
                        tf.contrib.summary.image('z_std', extras['z_std'][tf.newaxis,...,tf.newaxis])
        """

        global_step += 1


if __name__=='__main__':
    main()
