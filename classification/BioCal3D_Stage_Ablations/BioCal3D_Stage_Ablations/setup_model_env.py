"""Install and check one encoder in its dedicated BioCal3D environment."""
import argparse
import hashlib
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
SETTINGS = {
    "dinov3": {
        "python": (3, 11),
        "torch": "2.7.1",
        "torchvision": "0.22.1",
        "cuda": "12.6",
        "wheel_index": "cu126",
        "repo": "https://github.com/facebookresearch/dinov3.git",
        "checkpoint": "dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    },
    "concerto": {
        "python": (3, 10),
        "torch": "2.5.0",
        "torchvision": "0.20.0",
        "cuda": "12.4",
        "wheel_index": "cu124",
        "repo": "https://github.com/Pointcept/Concerto.git",
        "checkpoint": "concerto_small.pth",
    },
}
CONCERTO_URL = "https://huggingface.co/Pointcept/Concerto/resolve/main/concerto_small.pth"
CONCERTO_SHA256 = "251441d92ac7d9cff9e0c3ae22bdf2dc416d1c9dbb7a75f1922c08bfd1a18868"
SCATTER_WHEELS = "https://data.pyg.org/whl/torch-2.5.0+cu124.html"


def run(*args, **kwargs):
    return subprocess.run(args, check=True, **kwargs)


def pip_install(*args):
    run(sys.executable, "-m", "pip", "install", *args)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else sha256_stream(stream)


def sha256_stream(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def concerto_checkpoint(destination):
    if destination.is_file():
        if sha256(destination) != CONCERTO_SHA256:
            raise ValueError(f"Unexpected SHA256 for {destination}. Use the official Concerto Small file.")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading the official Concerto Small checkpoint.", flush=True)
    with TemporaryDirectory(dir=destination.parent) as temp:
        partial = Path(temp) / destination.name
        with urlopen(CONCERTO_URL, timeout=60) as response, partial.open("wb") as stream:
            shutil.copyfileobj(response, stream, length=1024 * 1024)
        if sha256(partial) != CONCERTO_SHA256:
            raise ValueError("The Concerto checkpoint did not match the expected SHA256.")
        partial.replace(destination)


def setup(args):
    model = SETTINGS[args.encoder]
    env = f"biocal3d-{args.encoder}"
    if os.environ.get("CONDA_DEFAULT_ENV") != env or sys.version_info[:2] != model["python"]:
        major, minor = model["python"]
        raise ValueError(f"Use the dedicated environment. Run conda create -n {env} python={major}.{minor} -y, "
                         f"then conda run -n {env} python setup_model_env.py {args.encoder}.")
    if platform.system() not in {"Windows", "Linux"} or platform.machine().lower() not in {"amd64", "x86_64"}:
        raise ValueError("These wheel combinations are for x64 Windows and Linux, including WSL2.")
    if not shutil.which("git"):
        raise ValueError("Install Git and reopen the terminal so git is available on PATH.")

    repo = ROOT / "vendor" / args.encoder
    if repo.exists():
        origin = run("git", "-C", str(repo), "remote", "get-url", "origin",
                     capture_output=True, text=True).stdout.strip()
        if origin.rstrip("/").removesuffix(".git").casefold() != model["repo"].removesuffix(".git").casefold():
            raise ValueError(f"Unexpected Git origin in {repo}. Check the repository before using it.")
    else:
        repo.parent.mkdir(parents=True, exist_ok=True)
        run("git", "clone", "--depth", "1", model["repo"], str(repo))

    pip_install(f'torch=={model["torch"]}', f'torchvision=={model["torchvision"]}',
                "--index-url", f'https://download.pytorch.org/whl/{model["wheel_index"]}')
    if args.encoder == "concerto":
        pip_install("-r", str(ROOT / "requirements.txt"), "numpy==1.26.4", "addict", "timm",
                    "huggingface_hub", "einops", "psutil", "opencv-python-headless<4.12",
                    "spconv-cu124==2.3.8", "--only-binary", "spconv-cu124,cumm-cu124")
        pip_install("torch-scatter==2.1.2+pt25cu124", "--no-index", "--only-binary", ":all:",
                    "--find-links", SCATTER_WHEELS)
    else:
        pip_install("-r", str(ROOT / "requirements.txt"))

    run(sys.executable, "-m", "pip", "check")
    check = """
import sys
import torch
import torchvision
assert torch.version.cuda == sys.argv[1], (torch.__version__, torch.version.cuda)
if not torch.cuda.is_available():
    raise RuntimeError("PyTorch cannot see a CUDA GPU in this environment. Check nvidia-smi and the Windows driver.")
x = torch.ones(4, device="cuda")
assert x.sum().item() == 4
if sys.argv[2] == "concerto":
    sys.path.insert(0, sys.argv[3])
    from concerto import model, transform
    import spconv.pytorch
    from torch_scatter import scatter
    groups = torch.tensor([0, 0, 1, 1], device="cuda")
    assert scatter(x, groups, reduce="sum").tolist() == [2, 2]
print(f"CUDA check passed on {torch.cuda.get_device_name(0)} (PyTorch {torch.__version__}).")
"""
    run(sys.executable, "-c", check, model["cuda"], args.encoder, str(repo))

    checkpoint = args.checkpoint.resolve() if args.checkpoint else ROOT / "weights" / model["checkpoint"]
    if args.encoder == "concerto":
        if args.checkpoint and not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        concerto_checkpoint(checkpoint)
    elif not checkpoint.is_file():
        if args.checkpoint:
            raise FileNotFoundError(checkpoint)
        print(f"Environment checked. Download the official DINOv3 ViT-S/16 .pth file from Meta and save it as:\n{checkpoint}")
        return
    print(f"Encoder checkout: {repo}\nCheckpoint: {checkpoint}")
    print("Next: run the encoder's one-scan --smoke command from the README.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("encoder", choices=tuple(SETTINGS))
    parser.add_argument("--checkpoint", type=Path, help="Path to an existing official checkpoint.")
    args = parser.parse_args()
    try:
        setup(args)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(2, f"Setup stopped: {exc}\n")
