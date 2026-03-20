#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


REQUIRED_FEATURES = (
    "observation.images.front",
    "observation.state",
    "action",
)
REQUIRED_OBSERVATION_FEATURES = (
    "observation.images.front",
    "observation.state",
)
EXPECTED_STATE_DIM = 4
EXPECTED_ACTION_DIM = 2


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def _shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return [int(dim) for dim in shape]


def _dtype(value: Any) -> str:
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        return type(value).__name__
    return str(dtype)


def _resolve_checkpoint(path: Path) -> Path:
    path = path.expanduser().resolve()
    pretrained_model_dir = path / "pretrained_model"
    if pretrained_model_dir.is_dir():
        return pretrained_model_dir
    return path


def _resolve_policy_ref(policy: str) -> str:
    policy_path = Path(policy).expanduser()
    if policy_path.exists():
        return str(_resolve_checkpoint(policy_path))
    return policy


def _import_runtime():
    try:
        import torch
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing runtime dependency. Use a Python 3.12-3.13 environment with "
            "`pip install lerobot`, then rerun this command."
        ) from exc

    return torch, LeRobotDataset, DiffusionPolicy


def _import_hub():
    try:
        from huggingface_hub import HfApi
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing Hub dependency. Install the repo requirements first, then rerun."
        ) from exc

    return HfApi


def _load_dataset(repo_id: str, root: Path | None):
    torch, LeRobotDataset, DiffusionPolicy = _import_runtime()
    dataset_dir = None
    if root is not None:
        dataset_dir = root / repo_id
        if not dataset_dir.is_dir():
            raise SystemExit(f"Expected dataset directory at {dataset_dir}")
        dataset = LeRobotDataset(repo_id, root=root)
    else:
        dataset = LeRobotDataset(repo_id)

    if len(dataset) == 0:
        dataset_desc = str(dataset_dir) if dataset_dir is not None else repo_id
        raise SystemExit(f"Dataset is empty: {dataset_desc}")

    return torch, dataset, dataset_dir, DiffusionPolicy


def _assert_expected_shapes(sample: dict[str, Any]) -> None:
    state_shape = _shape(sample["observation.state"])
    action_shape = _shape(sample["action"])

    if not state_shape or state_shape[-1] != EXPECTED_STATE_DIM:
        raise SystemExit(
            "Unexpected `observation.state` shape. "
            f"Expected trailing dimension {EXPECTED_STATE_DIM}, got {state_shape}."
        )

    if not action_shape or action_shape[-1] != EXPECTED_ACTION_DIM:
        raise SystemExit(
            "Unexpected `action` shape. "
            f"Expected trailing dimension {EXPECTED_ACTION_DIM}, got {action_shape}."
        )


def verify_dataset(args: argparse.Namespace) -> None:
    root = Path(args.root).expanduser().resolve() if args.root else None
    _, dataset, dataset_dir, _ = _load_dataset(args.repo_id, root)
    sample = dataset[0]

    missing = [key for key in REQUIRED_FEATURES if key not in sample]
    if missing:
        raise SystemExit(
            "Dataset is missing required features: " + ", ".join(missing)
        )

    _assert_expected_shapes(sample)

    info = getattr(dataset, "info", {}) or {}
    camera_keys = list(getattr(dataset, "camera_keys", []))

    print(f"dataset_source={'local' if dataset_dir is not None else 'hub'}")
    if dataset_dir is not None:
        print(f"dataset_dir={dataset_dir}")
    print(f"repo_id={args.repo_id}")
    print(f"num_samples={len(dataset)}")
    if "fps" in info:
        print(f"fps={info['fps']}")
    if "video" in info:
        print(f"video={info['video']}")
    if camera_keys:
        print(f"camera_keys={camera_keys}")
    print("required_features:")
    for key in REQUIRED_FEATURES:
        value = sample[key]
        print(f"  {key}: shape={_shape(value)} dtype={_dtype(value)}")


def _to_policy_value(torch_module: Any, value: Any, device: Any) -> Any:
    if hasattr(value, "to"):
        return value.to(device)
    if isinstance(value, (list, tuple)):
        return torch_module.as_tensor(value, device=device)
    return value


def infer_checkpoint(args: argparse.Namespace) -> None:
    root = Path(args.root).expanduser().resolve() if args.root else None
    policy_ref = _resolve_policy_ref(args.policy)
    torch, dataset, _, DiffusionPolicy = _load_dataset(args.repo_id, root)

    index = int(args.index)
    if index < 0 or index >= len(dataset):
        raise SystemExit(
            f"Index {index} is out of range for dataset with {len(dataset)} samples."
        )

    sample = dataset[index]
    missing = [key for key in REQUIRED_OBSERVATION_FEATURES if key not in sample]
    if missing:
        raise SystemExit(
            "Dataset sample is missing required observation features: "
            + ", ".join(missing)
        )

    _assert_expected_shapes(sample)

    device = torch.device(args.device)
    policy = DiffusionPolicy.from_pretrained(policy_ref)
    policy.eval()
    policy.to(device)
    if hasattr(policy, "reset"):
        policy.reset()

    observation = {
        key: _to_policy_value(torch, sample[key], device)
        for key in REQUIRED_OBSERVATION_FEATURES
    }

    with torch.inference_mode():
        action = policy.select_action(observation)

    if hasattr(action, "detach"):
        action = action.detach().cpu()

    print(f"policy_ref={policy_ref}")
    print(f"dataset_index={index}")
    print(f"predicted_action_shape={_shape(action)}")
    print(f"predicted_action={action}")


def push_folder_to_hub(args: argparse.Namespace) -> None:
    folder = Path(args.folder).expanduser().resolve()
    if not folder.is_dir():
        raise SystemExit(f"Expected folder to upload at {folder}")

    HfApi = _import_hub()
    api = HfApi()
    _ = api.whoami()
    api.create_repo(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        private=args.private,
        exist_ok=True,
    )
    api.upload_folder(
        folder_path=str(folder),
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        commit_message=args.commit_message,
    )

    repo_prefix = "datasets" if args.repo_type == "dataset" else ""
    if repo_prefix:
        print(f"https://huggingface.co/{repo_prefix}/{args.repo_id}")
    else:
        print(f"https://huggingface.co/{args.repo_id}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify and smoke-test the local Rotom LeRobot dataset."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify = subparsers.add_parser(
        "verify-dataset",
        help="Load the local dataset and confirm the expected Rotom features exist.",
    )
    verify.add_argument("--repo-id", default="local/rotom_task_teleop")
    verify.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parents[1] / "data"),
        help="Dataset root for local mode. Omit this flag to load directly from the Hub.",
    )
    verify.set_defaults(func=verify_dataset)

    infer = subparsers.add_parser(
        "infer-checkpoint",
        help="Run one offline action prediction from a trained diffusion policy.",
    )
    infer.add_argument("--repo-id", default="local/rotom_task_teleop")
    infer.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parents[1] / "data"),
        help="Dataset root for local mode. Omit this flag to load directly from the Hub.",
    )
    infer.add_argument(
        "--policy",
        "--checkpoint",
        dest="policy",
        default=str(
            Path(__file__).resolve().parents[1]
            / "outputs/train/rotom_task_teleop_diffusion/checkpoints/last/pretrained_model"
        ),
        help="Local pretrained_model directory or a Hub model repo id.",
    )
    infer.add_argument(
        "--device",
        default="mps",
    )
    infer.add_argument("--index", default=0, type=int)
    infer.set_defaults(func=infer_checkpoint)

    push = subparsers.add_parser(
        "push-folder-to-hub",
        help="Create or update a Hub repo from a local folder.",
    )
    push.add_argument("--repo-id", required=True)
    push.add_argument("--folder", required=True)
    push.add_argument("--repo-type", choices=("dataset", "model"), required=True)
    push.add_argument("--private", type=_parse_bool, default=True)
    push.add_argument("--commit-message", default="Upload from diffusion-policy")
    push.set_defaults(func=push_folder_to_hub)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
