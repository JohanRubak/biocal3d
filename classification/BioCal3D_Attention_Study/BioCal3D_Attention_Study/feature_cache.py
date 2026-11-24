"""Extract compact bags from frozen DINOv3 or Concerto encoders."""
import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from harmonized_inputs import harmonized_inputs, paired_mesh
from images_2d import render, to_grayscale
from models_2d import Encoder, commit
from study_data import digest, geometry, make_surface, seed_for, signature, write_json
from study_io import read_json, save_npz
from study_protocol import INPUTS, input_modes, is_lead

IMAGE_SIZE = 448
FIELD_MM = 13.4
BAG_SIZE = 512
SURFACE = dict(align_plane=True, scale_mm=10.0, sampling_seed=123, sampling="surface",
               points=8192, inputs="geometry_rgb")
VOXEL_MM = .10
ATTENTION_PATCH = 128
ROOT = Path(__file__).resolve().parent
# These revisions produce identical encoder inputs and bags for existing modes.
COMPATIBLE_SOURCES = {"3fcc064fa3cee6de0b77c2f68f54e30eb09e8a52ade7b3381cce97e7f401de0f",
                      "05e90655d24c3c38a2b83ebae68a625c9cdb5ca80118ee2c8ab38919e1a6ee11"}


def bag_path(run, mode, row):
    return run / "features" / mode / f"{row['row_id']}.npz"


def view_path(run, row, harmonized=False):
    suffix = "_harmonized" if harmonized else ""
    return run / "views" / f"{row['row_id']}{suffix}.npz"


def details_path(run, mode, row):
    return run / "details" / mode / f"{row['row_id']}.npz"


def mode_settings(run, mode):
    return read_json(run / "features" / mode / "settings.json")


def key_for(settings, row):
    return signature([settings, row["row_id"], row["geometry_sha256"], is_lead(row)])


def cache_ready(path, key):
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            if data["key"].item() != key:
                raise ValueError(f"Cache belongs to a different run: {path}")
    except (OSError, KeyError) as exc:
        raise ValueError(f"Unreadable cache: {path}. Remove this file and retry.") from exc
    return True


def save_bag(path, key, row, native, spatial, details=None):
    if native.ndim == 3:
        coverage = spatial["coverage"]
        tokens = native.reshape(native.shape[0], -1).T
        indices = np.flatnonzero(coverage.ravel() >= .05)
    else:
        tokens = native
        indices = np.arange(len(tokens))
    if not len(indices) or not np.isfinite(tokens).all():
        raise ValueError(f"Invalid spatial features for {row['scan_id']}.")
    if len(indices) > BAG_SIZE:
        rng = np.random.default_rng(seed_for(row["row_id"], 751))
        indices = np.sort(rng.choice(indices, BAG_SIZE, replace=False))
    features = np.ascontiguousarray(tokens[indices], dtype=np.float32)
    rms = np.sqrt(np.einsum("ij,ij->i", features, features) / features.shape[1])
    features /= np.maximum(rms[:, None], 1e-6)
    arrays = {}
    if is_lead(row) or details is not None:
        arrays.update(spatial, native=np.asarray(native, dtype=np.float32), indices=indices)
    if details is None:
        save_npz(path, key=np.array(key), bag=features, **arrays)
    else:
        with np.load(path, allow_pickle=False) as cached:
            reference = cached["bag"]
            if reference.shape != features.shape or not np.allclose(reference, features, rtol=2e-3, atol=2e-4):
                raise ValueError(f"Replayed features differ from the fitted classifier input: {row['scan_id']}. "
                                 "Use the encoder checkout and environment of the original run.")
        save_npz(details, key=np.array(key), **arrays)


def corrected_mesh(mesh, scanner, transform):
    rgb = np.asarray(mesh[3], dtype=np.float32)
    new = np.asarray(transform(rgb.copy(), scanner), dtype=np.float32)
    if new.shape != rgb.shape or not np.isfinite(new).all() or new.min() < 0 or new.max() > 1:
        raise ValueError("Harmonisation must return finite vertex RGB in [0, 1], with the original shape.")
    return (*mesh[:3], new, *mesh[4:])


def extract(run, family, repo, checkpoint, artifact=None, device="cuda", examples_file=None, *, harmonized_roi=None):
    import torch

    modes = input_modes(artifact is not None or harmonized_roi is not None, family)
    rows = read_json(run / "cohort.json")["rows"]
    if examples_file is not None:
        examples = read_json(examples_file)
        requested = {entry["scan_id"] for entry in examples["examples"]}
        rows = [row for row in rows if row["role"] == "test" and row["scan_id"] in requested]
        if {row["scan_id"] for row in rows} != requested:
            raise ValueError("The illustration selections must belong to this run's held-out test split.")
    if not checkpoint.is_file():
        raise ValueError(f"Checkpoint missing: {checkpoint}")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"The biocal3d-{family} environment cannot access CUDA.")
    transform, harmonizer = None, None
    if harmonized_roi is not None:
        corrected_rows, harmonizer = harmonized_inputs(run, harmonized_roi)
        harmonizer["pairing_code_sha256"] = digest(ROOT / "harmonized_inputs.py")
    elif artifact is not None:
        from colour_harmonization import load_transform
        if not artifact.is_file():
            raise ValueError(f"Harmonisation artefact missing: {artifact}")
        transform = load_transform(artifact)
        if not callable(transform):
            raise ValueError("load_transform(artifact) must return a callable.")
        harmonizer = dict(artifact_sha256=digest(artifact),
                          adapter_sha256=digest(ROOT / "colour_harmonization.py"))
    common = dict(checkpoint_sha256=digest(checkpoint), repository_commit=commit(repo),
                  torch=str(torch.__version__), numpy=np.__version__, device=device, bag_size=BAG_SIZE,
                  code={name: digest(ROOT / name) for name in
                        ("feature_cache.py", "study_data.py", "images_2d.py", "models_2d.py", "concerto_backend.py")})
    spec = dict(image_size=IMAGE_SIZE, field_mm=FIELD_MM) if family == "dinov3" else dict(
        surface=SURFACE, voxel_mm=VOXEL_MM, attention_patch=ATTENTION_PATCH)
    settings = {}
    for mode in modes:
        variant = INPUTS[mode][1]
        data = dict(common, encoder=family, variant=variant, **spec)
        if variant == "harmonized":
            data["harmonizer"] = harmonizer
        path = run / "features" / mode / "settings.json"
        if path.is_file():
            saved = read_json(path)
            expected = data
            saved_source = saved["code"].get("feature_cache.py")
            if saved_source in COMPATIBLE_SOURCES:
                expected = dict(data, code=dict(data["code"], **{"feature_cache.py": saved_source}))
            if saved != expected:
                raise ValueError(f"{mode} settings or checkpoint changed. Choose a fresh --run folder.")
            data = saved
        elif examples_file is not None:
            raise ValueError(f"Run the experiment before extracting illustrations for {mode}.")
        else:
            write_json(path, data)
        settings[mode] = data

    encoder = None
    computed = 0
    for i, row in enumerate(rows, 1):
        keys = {mode: key_for(settings[mode], row) for mode in modes}
        destinations = {}
        for mode in modes:
            path = bag_path(run, mode, row)
            ready = cache_ready(path, keys[mode])
            if examples_file is not None:
                if not ready:
                    raise ValueError(f"Missing classifier input for the selected illustration: {path}")
                extra = details_path(run, mode, row)
                if not is_lead(row) and not cache_ready(extra, keys[mode]):
                    destinations[mode] = extra
            elif not ready:
                destinations[mode] = None
        pending = list(destinations)
        illustrated = family == "dinov3" and (is_lead(row) or examples_file is not None)
        need_view = illustrated and not cache_ready(view_path(run, row), keys["dinov3_rgb"])
        need_harmonized_view = (illustrated and harmonized_roi is not None
                                and not cache_ready(view_path(run, row, True), keys["dinov3_harmonized"]))
        if not pending and not need_view and not need_harmonized_view:
            continue
        if digest(row["mesh_path"]) != row["geometry_sha256"]:
            raise ValueError(f"Mesh changed after the split was frozen: {row['scan_id']}.")
        mesh = geometry(row)
        if mesh[3] is None or mesh[3].max() <= 0:
            raise ValueError(f"Missing vertex RGB: {row['scan_id']}")
        harmonized = None
        if need_harmonized_view or any(INPUTS[m][1] == "harmonized" for m in pending):
            harmonized = (paired_mesh(row, corrected_rows[row["row_id"]], mesh) if harmonized_roi is not None
                          else corrected_mesh(mesh, row["scanner"], transform))
        if family == "dinov3":
            raw, ids, _, meta = render(row, IMAGE_SIZE, FIELD_MM, mesh=mesh)
            roi = ids >= 0
            if need_view:
                save_npz(view_path(run, row), key=np.array(keys["dinov3_rgb"]), rgb=raw, roi=roi,
                         center=np.asarray(meta["center_mm"]), field=np.array(FIELD_MM))
            if harmonized is not None:
                corrected, corrected_ids, _, _ = render(row, IMAGE_SIZE, FIELD_MM, mesh=harmonized)
                if not np.array_equal(ids, corrected_ids):
                    raise RuntimeError(f"Harmonisation changed the render geometry for {row['scan_id']}.")
                if need_harmonized_view:
                    save_npz(view_path(run, row, True), key=np.array(keys["dinov3_harmonized"]), rgb=corrected)
            if encoder is None and pending:
                encoder = Encoder(family, {family: dict(repo=str(repo), checkpoint=str(checkpoint))}, device)
            for mode in pending:
                variant = INPUTS[mode][1]
                if variant == "gray":
                    image = to_grayscale(raw)
                elif variant == "harmonized":
                    image = corrected
                else:
                    image = raw
                _, native, coverage = encoder.extract(image, roi)
                save_bag(bag_path(run, mode, row), keys[mode], row, native, dict(coverage=coverage),
                         details=destinations[mode])
        else:
            if encoder is None:
                from concerto_backend import ConcertoEncoder
                encoder = ConcertoEncoder(SimpleNamespace(repo=repo, checkpoint=checkpoint,
                    grid_size=VOXEL_MM / SURFACE["scale_mm"], attention_patch=ATTENTION_PATCH), device)
            surface = make_surface(row, SURFACE, family, mesh=mesh)
            corrected = make_surface(row, SURFACE, family, mesh=harmonized) if harmonized is not None else None
            grid = np.floor((surface["coord"] - surface["grid_origin"]) /
                            (VOXEL_MM / SURFACE["scale_mm"]))
            if len(np.unique(grid, axis=0)) < 32:
                raise ValueError(f"Too few occupied Concerto voxels in {row['scan_id']}. Check units and ROI.")
            xyz = (surface["coord"] * SURFACE["scale_mm"]) @ surface["rotation"].T + surface["center_mm"]
            for mode in pending:
                variant = INPUTS[mode][1]
                source = corrected if variant == "harmonized" else surface
                if corrected is not None and not np.array_equal(corrected["coord"], surface["coord"]):
                    raise RuntimeError(f"Harmonisation changed sampled points for {row['scan_id']}.")
                inputs = {key: torch.from_numpy(source[key])[None] for key in
                          ("coord", "color", "normal", "grid_origin")}
                torch.manual_seed(seed_for(row["row_id"]))
                with torch.inference_mode():
                    features, _ = encoder(inputs, use_color=variant != "geometry")
                native = features[0].float().cpu().numpy()
                if native.ndim != 2 or len(native) != len(xyz):
                    raise ValueError(f"Concerto output does not match the sampled points for {row['scan_id']}.")
                save_bag(bag_path(run, mode, row), keys[mode], row, native, dict(xyz=xyz),
                         details=destinations[mode])
        computed += 1
        if computed == 1 or computed % 16 == 0:
            print(f"{family}: features checked through {i}/{len(rows)} scans", flush=True)
    print(f"{family}: {len(rows)} scans ready ({computed} computed in this invocation).", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--encoder", choices=("dinov3", "concerto"), required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    colour = parser.add_mutually_exclusive_group()
    colour.add_argument("--harmonization-artifact", type=Path)
    colour.add_argument("--harmonized-roi", type=Path)
    parser.add_argument("--examples-file", type=Path, help="Replay only selected held-out scan details.")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    try:
        extract(args.run, args.encoder, args.repo, args.checkpoint, args.harmonization_artifact,
                args.device, args.examples_file, harmonized_roi=args.harmonized_roi)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.exit(2, f"Error: {exc}\n")
