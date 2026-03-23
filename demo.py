
import argparse, torch, os, json
import shutil
import numpy as np
import mmcv
from mmcv import Config
from collections import OrderedDict

from nuscenes import NuScenes

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import pickle
from tqdm import tqdm

from visualization.dataset import ImagePoint_NuScenes_vis, DatasetWrapper_NuScenes_vis


def revise_ckpt(state_dict):
    tmp_k = list(state_dict.keys())[0]
    if tmp_k.startswith('module.'):
        state_dict = OrderedDict(
            {k[7:]: v for k, v in state_dict.items()})
    return state_dict

@torch.no_grad()
def main(args):

    cfg = Config.fromfile(args.py_config)
    dataset_config = cfg.dataset_params

    if cfg.get('occupancy', False):
        from builder import tpv_occupancy_builder as model_builder
    else:
        from builder import tpv_lidarseg_builder as model_builder
    
    my_model = model_builder.build(cfg.model).to(device)
    if args.ckpt_path:
        ckpt = torch.load(args.ckpt_path, map_location='cpu')
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        print(my_model.load_state_dict(revise_ckpt(ckpt),strict=False))
    my_model.eval()

    if args.vis_train:
        pkl_path = '/c20250502/wangyushen/Datasets/NuScenes/method/tpvformer/nuscenes_infos_train.pkl'
    else:
        pkl_path = '/c20250502/wangyushen/Datasets/NuScenes/method/tpvformer/nuscenes_infos_val.pkl'


    data_path = '/c20250502/wangyushen/Datasets/NuScenes/v1.0-trainval/'
    label_mapping = dataset_config['label_mapping']

    nusc = NuScenes(version='v1.0-trainval', dataroot=data_path, verbose=True)

    pt_dataset = ImagePoint_NuScenes_vis(
        data_path, imageset=pkl_path,
        label_mapping=label_mapping, nusc=nusc)

    dataset = DatasetWrapper_NuScenes_vis(
        pt_dataset,
        grid_size=cfg.grid_size,
        fixed_volume_space=dataset_config['fixed_volume_space'],
        max_volume_space=dataset_config['max_volume_space'],
        min_volume_space=dataset_config['min_volume_space'],
        ignore_label=dataset_config["fill_label"],
        phase='val'
    )
    print(len(dataset))
    
    # Occ感知
    voxel_origin = dataset_config['min_volume_space']
    voxel_max = dataset_config['max_volume_space']
    grid_size = cfg.grid_size
    resolution = [(e - s) / l for e, s, l in zip(voxel_max, voxel_origin, grid_size)]
    print('voxel_origin: ', voxel_origin)
    print('voxel_max: ', voxel_max)
    print('grid_size: ', grid_size)
    print('resolution: ', resolution)
    
    save_root = args.save_path
    os.makedirs(save_root, exist_ok=True)

    num_frames = len(dataset)
    for index in tqdm(range(num_frames)):
        print(f'processing frame {index}')
        batch_data, filelist, scene_meta, timestamp = dataset[index]
        imgs, img_metas, vox_label, grid, pt_label = batch_data # vox_label: (100 100 8) grid: (34720 3) pt_label: (34720 1)
        imgs = torch.from_numpy(np.stack([imgs]).astype(np.float32)).to(device)
        grid = torch.from_numpy(np.stack([grid]).astype(np.float32)).to(device)        

        outputs_vox, outputs_pts = my_model(img=imgs, img_metas=[img_metas], points=grid.clone()) # outputs_vox: (1 18 100 100 8) output_pts: (1 18 34720 1 1) 

        predict_vox = torch.argmax(outputs_vox, dim=1) # bs, w, h, z
        predict_vox = predict_vox.squeeze(0).cpu().numpy() # w, h, z # (100 100 8)
        predict_pts = torch.argmax(outputs_pts, dim=1) # bs, n, 1, 1
        predict_pts = predict_pts.squeeze().cpu().numpy() # n (34720,)
        
        results = dict(
            occ_pred = predict_vox,     # (100 100 8)
            occ_gt = vox_label,         # (100 100 8)
            pt_pred = predict_pts,      # (num_pt,)
            pt_gt = pt_label.flatten(), # (num_pt,)
            grid = grid.squeeze(0).cpu().numpy()                 # (num_pt,3)
        )
        save_name = scene_meta['name']+"_"+scene_meta['token']
        save_path = os.path.join(save_root, f'{save_name}.pkl')
        with open(save_path, "wb") as f:
            pickle.dump(results,f)
    return



if __name__ == "__main__":
    device = torch.device('cuda:0')

    parser = argparse.ArgumentParser(description='TPVFormer Demo.')
    parser.add_argument('--py-config', default='config/tpv04_occupancy.py')
    parser.add_argument('--vis-train', action='store_true', default=False)
    parser.add_argument('--work-dir', type=str, default='out/tpv_occupancy')
    parser.add_argument('--save-path', type=str, default='out/tpv_occupancy/frames')
    parser.add_argument('--ckpt-path', type=str, default='out/tpv_occupancy/latest.pth')
    args = parser.parse_args()
    print(args)

    main(args)



