from dataclasses import dataclass
from typing import Optional

from hydra.core.config_store import ConfigStore

# from aris.config.object_config import ObjectConfig
# from aris.config.scene_config import SceneConfig


@dataclass
class NerfRenderConfig:
    # scene: NerfSceneConfig

    scale: float
    device: str
    config: Optional[str]

    # TODO: config file!!!

    expname: str    # experiment name
    basedir: str    # where to store ckpts and logs
    datadir: str    # input data directory

    ### training options
    netdepth: int       # layers in network
    netwidth: int       # channels per layer
    netdepth_fine: int  # layers in fine network
    netwidth_fine: int  # channels per layer fine network
    N_rand: int         # batch size (number of random rays per gradient step)
    lrate: float        # learning rate
    lrate_decay: int    # exponential learning rate decay (in 1000 steps)
    chunk: int          # number of rays processed in parallel, decrease if running out of memory
    netchunk: int       # number of pts sent through network in parallel, decrease if running out of memory

    no_batching: bool   # only take random rays from 1 image at a time
    no_reload: bool     # do not reload weights from saved ckpt
    ft_path: str        # specific weights npy file to reload for coarse network

    ### rendering options
    N_samples: int      # number of coarse samples per ray
    N_importance: int   # number of additional fine samples per ray
    perturb: float      # set to 0. for no jitter, 1. for jitter
    use_viewdirs: bool  # use full 5D input instead of 3D
    i_embed: int        # set 0 for default positional encoding, -1 for none
    multires: int       # log2 of max freq for positional encoding (3D location)
    multires_views: int # log2 of max freq for positional encoding (2D direction)
    raw_noise_std: float# std dev of noise added to regularize sigma_a output, 1e0 recommended

    render_only: bool   # do not optimize, reload weights and render out render_poses path
    render_test: bool   # render the test set instead of render_poses path
    render_factor: int  # downsampling factor to speed up rendering, set 4 or 8 for fast preview

    ### training options (part 2)
    precrop_iters: int  # number of steps to train on central crops
    precrop_frac: float # fraction of img taken for central crops

    ### dataset options
    dataset_type: str   # options: llff / blender / deepvoxels
    testskip: int       # will load 1/N images from test/val sets, useful for large datasets like deepvoxels

    ### deepvoxels flags
    shape: str          # options : armchair / cube / greek / vase

    ### blender flags
    white_bkgd: bool    # set to render synthetic data on a white bkgd (always use for dvoxels)
    half_res: bool      # load blender synthetic data at 400x400 instead of 800x800

    ### llff flags
    factor: int         # downsample factor for LLFF images
    no_ndc: bool        # do not use normalized device coordinates (set for non-forward facing scenes)
    lindisp: bool       # sampling linearly in disparity rather than depth
    spherify: bool      # set for spherical 360 scenes
    llffhold: int       # will take every 1/N images as LLFF test set, paper uses 8

    ### logging/saving options
    i_print: int        # frequency of console printout and metric logging
    i_img: int          # frequency of tensorboard image logging
    i_weights: int      # frequency of weight ckpt saving
    i_testset: int      # frequency of testset saving
    i_video: int        # frequency of render_poses video saving

    # TODO: move some config into scenes/integrators???

    # scene: NerfSceneConfig
    # integrator: ObjectConfig

cs = ConfigStore.instance()
cs.store(name="nerf_render_schema", node=NerfRenderConfig)
