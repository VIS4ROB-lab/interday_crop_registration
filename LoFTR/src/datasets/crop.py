import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from loguru import logger

from src.utils.dataset import read_crop_gray, read_crop_depth, read_crop_height_map, cut_crop_height_map


class CropDataset(Dataset):
    def __init__(self,
                 root_dir,
                 npz_path,
                 mode='train',
                 min_overlap_score=0.4,
                 img_resize=None,
                 df=None,
                 img_padding=False,
                 depth_padding=False,
                 augment_fn=None,
                 img_crop=None,
                 compensate_height_diff=False,
                 crop_heightmap_pad=6000,
                 crop_heightmap_cut=2000,
                 **kwargs):
        """
        Manage one scene(npz_path) of Crop dataset.
        
        Args:
            root_dir (str): crop root directory that conatains different scenes.
            npz_path (str): {scene_id}.npz path. This contains image pair information of a scene.
            mode (str): options are ['train', 'val', 'test']
            min_overlap_score (float): how much a pair should have in common. In range of [0, 1]. Set to 0 when testing.
            img_resize (int, optional): the longer edge of resized images. None for no resize. 640 is recommended.
                                        This is useful during training with batches and testing with memory intensive algorithms.
            df (int, optional): image size division factor. NOTE: this will change the final image size after img_resize.
            img_padding (bool): If set to 'True', zero-pad the image to squared size. This is useful during training.
            depth_padding (bool): If set to 'True', zero-pad depthmap to (2000, 2000). This is useful during training.
            augment_fn (callable, optional): augments images with pre-defined visual effects.
        """
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode
        self.scene_id = npz_path.split('.')[0]

        # prepare scene_info and pair_info
        if mode == 'test' and min_overlap_score != 0:
            logger.warning("You are using `min_overlap_score`!=0 in test mode. Set to 0.")
            min_overlap_score = 0
        self.scene_info = dict(np.load(npz_path, allow_pickle=True))
        self.pair_infos = self.scene_info['pair_infos'].copy()
        del self.scene_info['pair_infos']
        self.pair_infos = [pair_info for pair_info in self.pair_infos if pair_info[1] > min_overlap_score]

        # parameters for image resizing, padding and depthmap padding
        if mode == 'train':
            assert img_resize is not None # and img_padding and depth_padding
        self.img_resize = img_resize
        self.df = df
        self.img_crop = img_crop

        # for training LoFTR
        self.augment_fn = augment_fn if mode == 'train' else None
        self.coarse_scale = getattr(kwargs, 'coarse_scale', 0.125)
        self.compensate_height_diff = compensate_height_diff

        # load height maps for later use
        if self.compensate_height_diff:
            self.height_maps = read_crop_height_map(self.scene_info['height_map_paths'].item(), pad_size=crop_heightmap_pad)
            self.crop_heightmap_cut = crop_heightmap_cut

        print('\ncompensate_height_diff:', self.compensate_height_diff)

    def __len__(self):
        return len(self.pair_infos)

    def __getitem__(self, idx):
        (idx0, idx1), overlap_score, pair_height_map_name = self.pair_infos[idx]
        height_map_name0, height_map_name1 = pair_height_map_name

        # read grayscale image and mask. (1, h, w) and (h, w)
        img_name0 = osp.join(self.root_dir, self.scene_info['image_paths'][idx0])
        img_name1 = osp.join(self.root_dir, self.scene_info['image_paths'][idx1])
        
        # TODO: Support augmentation & handle seeds for each worker correctly.
        image0, scale0 = read_crop_gray(
            img_name0, self.img_resize, self.df, None, self.img_crop)
            # np.random.choice([self.augment_fn, None], p=[0.5, 0.5]))
        image1, scale1 = read_crop_gray(
            img_name1, self.img_resize, self.df, None, self.img_crop)
            # np.random.choice([self.augment_fn, None], p=[0.5, 0.5]))

        # read intrinsics of original size
        K_0 = K_1 = torch.tensor(self.scene_info['intrinsics'].copy(), dtype=torch.float).reshape(3, 3)

        # read and compute relative poses
        T0 = self.scene_info['poses'][idx0]
        T1 = self.scene_info['poses'][idx1]
        T_0to1 = torch.tensor(np.matmul(np.linalg.inv(T1), T0), dtype=torch.float)[:4, :4]  # (4, 4)
        T_1to0 = T_0to1.inverse()


        # read depth. shape: (h, w)
        if self.mode in ['train', 'val']:
            depth0 = read_crop_depth(osp.join(self.root_dir, self.scene_info['depth_paths'][idx0]), 
                                     self.img_crop)
            depth1 = read_crop_depth(osp.join(self.root_dir, self.scene_info['depth_paths'][idx1]), 
                                     self.img_crop)
            
            if self.compensate_height_diff:
                height_map_name0, height_map_name1 = pair_height_map_name
                height_map0, height_map_info0 = cut_crop_height_map(self.height_maps[height_map_name0], T0, self.crop_heightmap_cut)
                height_map1, height_map_info1 = cut_crop_height_map(self.height_maps[height_map_name1], T1, self.crop_heightmap_cut)
            
        else:
            depth0 = depth1 = torch.tensor([])
            if self.compensate_height_diff:
                height_map0 = height_map1 = torch.tensor([])

        data = {
            'image0': image0,  # (1, h, w)
            'depth0': depth0,  # (h, w)
            'image1': image1,
            'depth1': depth1,
            'T_0to1': T_0to1,  # (4, 4)
            'T_1to0': T_1to0,
            'K0': K_0,  # (3, 3)
            'K1': K_1,
            'scale0': scale0,  # [scale_w, scale_h]
            'scale1': scale1,
            'dataset_name': 'Crop',
            'scene_id': self.scene_id,
            'pair_id': idx,
            'pair_names': (self.scene_info['image_paths'][idx0], self.scene_info['image_paths'][idx1]),
            }
        
        if self.compensate_height_diff:
            data.update({
                'compensate_height_diff': self.compensate_height_diff,
                'T0': torch.tensor(T0, dtype=torch.float),
                'T1': torch.tensor(T1, dtype=torch.float),
                'height_map0': height_map0,
                'height_map1': height_map1,
                'height_map_info0': height_map_info0,
                'height_map_info1': height_map_info1,
            })

        return data
