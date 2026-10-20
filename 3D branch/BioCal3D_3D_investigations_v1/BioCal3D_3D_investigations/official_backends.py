"""Official model adapters recovered from the user's existing BioCal3D benchmark.
No replacement backbone is supplied. Requires existing third_party repositories.
"""
from __future__ import annotations
import copy, hashlib, importlib, json, subprocess, sys, types
from pathlib import Path
import numpy as np
import torch
from torch import nn

def canonical_json(obj):
    return json.dumps(obj, sort_keys=True, default=str, allow_nan=False)

def signature(obj):
    return hashlib.sha256(canonical_json(obj).encode()).hexdigest()

def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def repo_info(path):
    result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else "not-a-git-checkout"

class TSGCEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        path = args.backends_root / "TSGCNet" / "TSGCNet.py"
        text = path.read_text(encoding="utf-8")
        # Compatibility patches for the official code's fixed CUDA/16k assumptions.
        replacements = {"device = torch.device('cuda')": "device = coor.device",
                        ".reshape(1,1,16000)": ".unsqueeze(1)",
                        "avgSum = avgSum_coor+avgSum_nor":
                        "avgSum = avgSum_coor+avgSum_nor\n        avgSum = torch.where(avgSum.abs() < 1e-6, torch.where(avgSum >= 0, torch.full_like(avgSum, 1e-6), torch.full_like(avgSum, -1e-6)), avgSum)"}
        for before, after in replacements.items():
            if before not in text:
                raise RuntimeError(f"Official TSGCNet source changed; compatibility patch missing: {before}")
            text = text.replace(before, after)
        module = types.ModuleType("biocal3d_official_tsgcnet")
        exec(compile(text, str(path), "exec"), module.__dict__)
        self.model = module.TSGCNet(k=args.neighbors, in_channels=12, output_channels=3)
        for p in self.model.pred4.parameters():
            p.requires_grad = False  # Original segmentation output is not the stage head.
        self.model.pred3.register_forward_hook(self._capture)
        self.latest = None
        self.channels = 128
        self.provenance = dict(source=str(path), source_sha256=file_hash(path),
                               commit=repo_info(path.parent), adaptation="pred3 local features + stage head; fixed-device/fixed-shape/numerical patches")

    def _capture(self, module, inputs, output):
        self.latest = output

    def forward(self, data):
        self.model(data["tsgc"].transpose(1, 2).contiguous())
        x = self.latest.transpose(1, 2)
        self.latest = None
        return x, data["coord"]

class PointNextEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        root = args.backends_root / "PointNeXt"
        sys.path.insert(0, str(root))
        from openpoints.utils import EasyConfig
        from openpoints.models import build_model_from_cfg
        cfg = EasyConfig()
        cfg.load(str(root / "cfgs" / "scanobjectnn" / "pointnext-s.yaml"), recursive=False)
        encoder_cfg = copy.deepcopy(cfg.model.encoder_args)
        encoder_cfg.in_channels = 9 if args.inputs == "geometry_rgb" else 6
        self.input_channels = encoder_cfg.in_channels
        self.model = build_model_from_cfg(encoder_cfg)
        self.channels = self.model.out_channels
        self.provenance = dict(commit=repo_info(root), openpoints_commit=repo_info(root / "openpoints"),
                               encoder_config=dict(encoder_cfg), adaptation="official S encoder + mean/max stage head; XYZ/normals/optional RGB")

    def forward(self, data):
        parts = [data["coord"], data["normal"]]
        if self.input_channels == 9:
            parts.append(data["color"] * 2 - 1)
        feat = torch.cat(parts, dim=-1).transpose(1, 2).contiguous()
        positions, features = self.model.forward_seg_feat(data["coord"].contiguous(), feat)
        return features[-1].transpose(1, 2), positions[-1]

class PretrainedEncoder(nn.Module):
    def __init__(self, name, args, device):
        super().__init__()
        sys.path.insert(0, str(args.backends_root / name))
        package = importlib.import_module(name)
        checkpoint = args.sonata_checkpoint if name == "sonata" else args.concerto_checkpoint
        model_name = str(checkpoint) if checkpoint else ("sonata" if name == "sonata" else args.concerto_variant)
        self.model = package.model.load(model_name, custom_config={"enable_flash": args.flash,
            "enc_patch_size": [args.attention_patch] * 5}).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad = False
        # Explicit voxel grid replaces scene-scale defaults and fragment voting.
        self.package = package
        self.normalize_color = package.transform.NormalizeColor()
        self.grid_size = args.grid_size
        self.provenance = dict(commit=repo_info(args.backends_root / name), model=model_name,
            checkpoint_sha256=file_hash(checkpoint) if checkpoint else None,
            initialization="official pretrained", grid_size=args.grid_size, flash=args.flash,
            adaptation="frozen encoder; deterministic voxel representatives; original-face mean/max pooling")

    def forward(self, data):
        if data["coord"].shape[0] != 1:
            raise ValueError("Pretrained feature extraction uses one scan at a time.")
        coord = data["coord"][0].detach().cpu().numpy()
        color = data["color"][0].detach().cpu().numpy()
        # The official transform owns normalization. Sonata currently uses
        # [0,255] -> [0,1]; do not assume the commonly used [-1,1] convention.
        color = self.normalize_color({"color": color * 255.})["color"]
        normal = data["normal"][0].detach().cpu().numpy()
        # Preserve the full-surface voxel origin during occlusion. Recomputing
        # minima after removing a patch would shift unrelated voxel assignments.
        origin = data['grid_origin'][0].detach().cpu().numpy() if 'grid_origin' in data else coord.min(0)
        grid = np.floor((coord - origin) / self.grid_size).astype(np.int32)
        _, first, inverse = np.unique(grid, axis=0, return_index=True, return_inverse=True)
        device = next(self.model.parameters()).device
        point = dict(coord=torch.from_numpy(coord[first]).to(device),
            grid_coord=torch.from_numpy(grid[first]).to(device),
            feat=torch.from_numpy(np.concatenate([coord[first], color[first], normal[first]], axis=1)).to(device),
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
