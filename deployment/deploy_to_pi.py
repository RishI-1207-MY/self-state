"""Cross-platform SSH/SCP Deployment Utility to transfer SafeStep & Optimized Models to Raspberry Pi 4.

Features:
- Creates a clean, lightweight deployment bundle (`deployment/safestep_rpi_bundle.tar.gz`) containing:
  - All SafeStep Python modules (`app.py`, `modules/`, `hardware/`, `utils/`, `training/`, `tests/`, `deployment/`)
  - Both configuration profiles (`config.yaml`, `config.windows.yaml`, `config.rpi.yaml`)
  - Trained & Raspberry Pi-optimized model weights (`best.pt`, `best.onnx`, `best_rpi_int8.onnx` for both Object and Currency models)
  - Evaluation reports, graphs, and `data.yaml` metadata (excludes raw multi-thousand-image training directories by default so transfer over Wi-Fi/LAN takes seconds rather than minutes; pass `--include-datasets` to transfer full datasets too)
- Transfers the bundle to the Raspberry Pi via `scp` and extracts it remotely via `ssh`
- Supports `--dry-run` (or `--bundle-only`) so you can package and verify the deployment archive on your Windows laptop before the Raspberry Pi is connected to the network.
"""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger import get_logger

logger = get_logger("SafeStep.Deploy")

ESSENTIAL_PATHS = [
    "app.py",
    "config.yaml",
    "config.windows.yaml",
    "config.rpi.yaml",
    "requirements.txt",
    "README.md",
    "modules",
    "hardware",
    "utils",
    "training",
    "tests",
    "deployment/install_raspberry_pi.sh",
    "deployment/deploy_to_pi.py",
    "deployment/RASPBERRY_PI_DEPLOYMENT_CHECKLIST.md",
    "models/object_detection/best.pt",
    "models/object_detection/best.onnx",
    "models/object_detection/best_rpi_int8.onnx",
    "models/object_detection/training_config.json",
    "models/currency_detection/best.pt",
    "models/currency_detection/best.onnx",
    "models/currency_detection/best_rpi_int8.onnx",
    "models/currency_detection/training_config.json",
    "models/easyocr",
    "datasets/README.md",
    "datasets/object_detection/data.yaml",
    "datasets/currency_detection/data.yaml",
    "datasets/dataset_generation",
    "evaluation",
]


def build_deployment_bundle(include_full_datasets: bool = False) -> Dict[str, Any]:
    """Create a compressed tar.gz archive ready for SCP transfer to Raspberry Pi 4."""
    bundle_dir = PROJECT_ROOT / "deployment"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = bundle_dir / "safestep_rpi_bundle.tar.gz"

    included_files: List[str] = []
    t0 = time.time()

    def _exclude_pycache(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo | None:
        if "__pycache__" in tarinfo.name or tarinfo.name.endswith(".pyc"):
            return None
        return tarinfo

    with tarfile.open(bundle_path, "w:gz") as tar:
        for rel_item in ESSENTIAL_PATHS:
            full_path = PROJECT_ROOT / rel_item
            if full_path.exists():
                tar.add(full_path, arcname=f"SafeStep/{rel_item}", filter=_exclude_pycache)
                included_files.append(rel_item)

        if include_full_datasets:
            for ds_name in ("object_detection", "currency_detection"):
                ds_path = PROJECT_ROOT / "datasets" / ds_name
                if ds_path.exists():
                    tar.add(ds_path, arcname=f"SafeStep/datasets/{ds_name}", filter=_exclude_pycache)
                    included_files.append(f"datasets/{ds_name}")
        else:
            # Include a small sample of held-out test images so the Raspberry Pi can run immediate self-tests
            for ds_name in ("object_detection", "currency_detection"):
                test_img_dir = PROJECT_ROOT / "datasets" / ds_name / "images" / "test"
                if test_img_dir.exists():
                    for sample_img in sorted(test_img_dir.glob("*.jpg"))[:10]:
                        rel_sample = sample_img.relative_to(PROJECT_ROOT).as_posix()
                        tar.add(sample_img, arcname=f"SafeStep/{rel_sample}")

    size_mb = round(bundle_path.stat().st_size / (1024 * 1024), 2)
    elapsed = round(time.time() - t0, 2)
    manifest = {
        "bundle_path": str(bundle_path),
        "bundle_size_mb": size_mb,
        "build_time_sec": elapsed,
        "included_top_level_items": included_files,
        "includes_full_datasets": include_full_datasets,
    }
    manifest_path = bundle_dir / "deployment_bundle_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Created Raspberry Pi deployment bundle: %s (%.2f MB)", bundle_path, size_mb)
    return manifest


def deploy_via_ssh_scp(
    host: str,
    user: str = "pi",
    port: int = 22,
    remote_dir: str = "~/SafeStep_Deploy",
    run_install: bool = False,
    dry_run: bool = False,
    include_full_datasets: bool = False,
) -> Dict[str, Any]:
    """Package SafeStep and transfer to Raspberry Pi via OpenSSH `scp` and `ssh`."""
    manifest = build_deployment_bundle(include_full_datasets=include_full_datasets)
    bundle_path = manifest["bundle_path"]

    scp_bin = shutil.which("scp")
    ssh_bin = shutil.which("ssh")
    target = f"{user}@{host}"

    ssh_mkdir_cmd = [ssh_bin or "ssh", "-p", str(port), target, f"mkdir -p {remote_dir}"]
    scp_cmd = [scp_bin or "scp", "-P", str(port), bundle_path, f"{target}:{remote_dir}/safestep_rpi_bundle.tar.gz"]
    remote_extract = f"cd {remote_dir} && tar -xzf safestep_rpi_bundle.tar.gz && chmod +x SafeStep/deployment/*.sh"
    if run_install:
        remote_extract += " && bash SafeStep/deployment/install_raspberry_pi.sh"
    ssh_extract_cmd = [ssh_bin or "ssh", "-p", str(port), target, remote_extract]

    if dry_run or not host or host == "DRY_RUN":
        logger.info("Dry-run / bundle-only mode completed. Ready to transfer when Raspberry Pi is online.")
        return {
            "status": "BUNDLE_READY_DRY_RUN",
            "bundle_manifest": manifest,
            "scp_command_to_run": " ".join(scp_cmd),
            "ssh_extract_command_to_run": " ".join(ssh_extract_cmd),
        }

    if not scp_bin or not ssh_bin:
        return {
            "status": "FAILED_MISSING_OPENSSH_CLIENT",
            "reason": "OpenSSH client (ssh/scp) not found in PATH.",
            "bundle_manifest": manifest,
        }

    logger.info("Connecting to Raspberry Pi at %s:%d ...", target, port)
    res_mkdir = subprocess.run(ssh_mkdir_cmd, capture_output=True, text=True, check=False)
    if res_mkdir.returncode != 0:
        return {
            "status": "FAILED_SSH_CONNECTION",
            "stderr": res_mkdir.stderr.strip(),
            "bundle_manifest": manifest,
        }

    logger.info("Transferring %s (%.2f MB) via SCP...", bundle_path, manifest["bundle_size_mb"])
    res_scp = subprocess.run(scp_cmd, capture_output=True, text=True, check=False)
    if res_scp.returncode != 0:
        return {
            "status": "FAILED_SCP_TRANSFER",
            "stderr": res_scp.stderr.strip(),
            "bundle_manifest": manifest,
        }

    logger.info("Extracting bundle on Raspberry Pi (%s)...", remote_dir)
    res_ext = subprocess.run(ssh_extract_cmd, capture_output=True, text=True, check=False)
    if res_ext.returncode != 0:
        return {
            "status": "FAILED_REMOTE_EXTRACT",
            "stderr": res_ext.stderr.strip(),
            "bundle_manifest": manifest,
        }

    return {
        "status": "DEPLOYED_SUCCESSFULLY",
        "target": f"{target}:{remote_dir}/SafeStep",
        "bundle_manifest": manifest,
        "stdout": res_ext.stdout.strip(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package and deploy SafeStep to Raspberry Pi 4 via SSH/SCP.")
    parser.add_argument("--host", type=str, default="", help="Raspberry Pi IP address or hostname (e.g. raspberrypi.local or 192.168.1.50)")
    parser.add_argument("--user", type=str, default="pi", help="Raspberry Pi SSH username")
    parser.add_argument("--port", type=int, default=22, help="SSH port")
    parser.add_argument("--remote-dir", type=str, default="~/SafeStep_Deploy", help="Target directory on Raspberry Pi")
    parser.add_argument("--run-install", action="store_true", help="Automatically run install_raspberry_pi.sh after transfer")
    parser.add_argument("--include-datasets", action="store_true", help="Include full raw/split dataset images in archive")
    parser.add_argument("--dry-run", action="store_true", help="Build deployment archive and print SSH/SCP commands without connecting")
    args = parser.parse_args()

    out = deploy_via_ssh_scp(
        host=args.host,
        user=args.user,
        port=args.port,
        remote_dir=args.remote_dir,
        run_install=args.run_install,
        dry_run=(args.dry_run or not bool(args.host)),
        include_full_datasets=args.include_datasets,
    )
    print(json.dumps(out, indent=2))
