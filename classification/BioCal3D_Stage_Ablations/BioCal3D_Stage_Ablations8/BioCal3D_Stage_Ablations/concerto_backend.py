"""Concerto adapter adapted from the existing BioCal3D 3D investigations."""
import importlib
import sys
import numpy as np
import torch
from torch import nn

class ConcertoEncoder(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        sys.path.insert(0, str(args.repo))
        package = importlib.import_module("concerto")
        checkpoint = args.checkpoint
        model_name = str(checkpoint)
        self.model = package.model.load(model_name, custom_config={"enable_flash": False,
            "enc_patch_size": [args.attention_patch] * 5}).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad = False
        # Explicit voxel grid replaces scene-scale defaults and fragment voting.
        self.normalize_color = package.transform.NormalizeColor()
        self.grid_size = args.grid_size

    def forward(self, data, use_color):
        if data["coord"].shape[0] != 1:
            raise ValueError("Pretrained feature extraction uses one scan at a time.")
        coord = data["coord"][0].detach().cpu().numpy()
        color = data["color"][0].detach().cpu().numpy()
        # The official transform owns normalization. Sonata currently uses
        # [0,255] -> [0,1]; do not assume the commonly used [-1,1] convention.
        color = self.normalize_color({"color": color * 255.})["color"]
        if not use_color:
            # The fixed absent-colour vector is zero in model input space.
            color = np.zeros_like(color)
        normal = data["normal"][0].detach().cpu().numpy()
        # Preserve the full-surface voxel origin during occlusion. Recomputing
        # minima after removing a patch would shift unrelated voxel assignments.
        origin = data['grid_origin'][0].detach().cpu().numpy() if 'grid_origin' in data else coord.min(0)
        grid = np.floor((coord - origin) / self.grid_size).astype(np.int32)
        _, first, inverse = np.unique(grid, axis=0, return_index=True, return_inverse=True)
        device = next(self.model.parameters()).device
        point = dict(coord=torch.from_numpy(coord[first]).to(device),
            grid_coord=torch.from_numpy(grid[first]).to(device),
            feat=torch.from_numpy(np.concatenate([coord[first], color[first], normal[first]], axis=1).astype(np.float32)).to(device),
            offset=torch.tensor([len(first)], dtype=torch.int64, device=device))
        point = self.model(point)
        # Official encoder hierarchy upcast: concatenate the first two parent
        # feature levels, then propagate to the encoder's input resolution.
        for _ in range(2):
            if "pooling_parent" not in point or "pooling_inverse" not in point:
                raise RuntimeError("Unexpected pretrained hierarchy; official API changed.")
            parent = point.pop("pooling_parent")
            idx = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[idx]], dim=-1)
            point = parent
        while "pooling_parent" in point:
            parent = point.pop("pooling_parent")
            idx = point.pop("pooling_inverse")
            parent.feat = point.feat[idx]
            point = parent
        feat = point.feat[torch.from_numpy(inverse).to(device)]
        return feat[None], data["coord"]
