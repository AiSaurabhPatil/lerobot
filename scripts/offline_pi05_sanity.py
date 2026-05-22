#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from peft import PeftConfig, PeftModel

from lerobot.configs import PreTrainedConfig
from lerobot.datasets import LeRobotDataset, resolve_delta_timestamps
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-repo-id", default="quest3-acone_v3")
    parser.add_argument("--dataset-root", default="datasets/quest3-acone_v3")
    parser.add_argument("--indices", nargs="+", type=int, default=[0, 500, 2000, 5000, 10000, 15000, 20000])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--output-json", default="/tmp/pi05_offline_sanity.json")
    return parser.parse_args()


def tensor_chw_uint8_to_hwc_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.permute(1, 2, 0).contiguous().numpy()


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)

    checkpoint = Path(args.checkpoint)
    policy_cfg = PreTrainedConfig.from_pretrained(checkpoint)
    policy_cfg.device = args.device
    policy_cfg.num_inference_steps = args.num_inference_steps

    policy_cls = get_policy_class(policy_cfg.type)
    if policy_cfg.use_peft:
        peft_cfg = PeftConfig.from_pretrained(checkpoint)
        policy = policy_cls.from_pretrained(peft_cfg.base_model_name_or_path, config=policy_cfg)
        policy = PeftModel.from_pretrained(policy, checkpoint, config=peft_cfg)
    else:
        policy = policy_cls.from_pretrained(checkpoint, config=policy_cfg)
    policy = policy.to(args.device).eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )

    meta_dataset = LeRobotDataset(args.dataset_repo_id, root=args.dataset_root, return_uint8=True)
    delta_timestamps = resolve_delta_timestamps(policy_cfg, meta_dataset.meta)
    dataset = LeRobotDataset(
        args.dataset_repo_id,
        root=args.dataset_root,
        delta_timestamps=delta_timestamps,
        return_uint8=True,
    )

    names = policy_cfg.action_feature_names or [f"dim_{i}" for i in range(policy_cfg.output_features["action"].shape[0])]
    records = []
    per_joint_abs = []
    checked_indices = [idx for idx in args.indices if 0 <= idx < len(dataset)]
    print(
        f"offline sanity: checkpoint={checkpoint} dataset={args.dataset_repo_id} "
        f"device={args.device} num_inference_steps={args.num_inference_steps} samples={checked_indices}",
        flush=True,
    )

    for idx in checked_indices:
        item = dataset[idx]
        obs = {
            "observation.state": item["observation.state"].numpy(),
            "observation.images.head_camera": tensor_chw_uint8_to_hwc_numpy(item["observation.images.head_camera"]),
            "observation.images.left_wrist_camera": tensor_chw_uint8_to_hwc_numpy(
                item["observation.images.left_wrist_camera"]
            ),
            "observation.images.right_wrist_camera": tensor_chw_uint8_to_hwc_numpy(
                item["observation.images.right_wrist_camera"]
            ),
        }
        batch = prepare_observation_for_inference(obs, torch.device(args.device), item["task"], "acone")
        with torch.inference_mode():
            processed_batch = preprocessor(batch)
            pred = policy.predict_action_chunk(processed_batch)
            pred = postprocessor(pred).squeeze(0).cpu()

        target = item["action"].float().cpu()
        steps = min(pred.shape[0], target.shape[0])
        abs_err = (pred[:steps] - target[:steps]).abs()
        per_joint_abs.append(abs_err.mean(dim=0).numpy())
        record = {
            "index": int(idx),
            "episode": int(item["episode_index"]),
            "frame": int(item["frame_index"]),
            "mean_abs": float(abs_err.mean()),
            "first_step_abs": float(abs_err[0].mean()),
            "pred0": {name: float(value) for name, value in zip(names, pred[0], strict=False)},
            "target0": {name: float(value) for name, value in zip(names, target[0], strict=False)},
        }
        records.append(record)
        print(
            f"idx={record['index']:5d} ep={record['episode']:2d} frame={record['frame']:4d} "
            f"mean_abs={record['mean_abs']:.4f} first_step_abs={record['first_step_abs']:.4f}",
            flush=True,
        )

    joint_summary = {}
    if per_joint_abs:
        joint_abs = np.stack(per_joint_abs).mean(axis=0)
        joint_summary = {name: float(value) for name, value in zip(names, joint_abs, strict=False)}

    result = {
        "checkpoint": str(checkpoint),
        "dataset": args.dataset_repo_id,
        "device": args.device,
        "num_inference_steps": args.num_inference_steps,
        "samples": records,
        "summary_mean_abs": float(np.mean([r["mean_abs"] for r in records])) if records else None,
        "summary_first_step_abs": float(np.mean([r["first_step_abs"] for r in records])) if records else None,
        "per_joint_mean_abs": joint_summary,
    }
    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {args.output_json}", flush=True)
    print("summary_mean_abs", result["summary_mean_abs"], flush=True)
    for name, value in sorted(joint_summary.items(), key=lambda item: -item[1]):
        print(f"{name:14s} mean_abs={value:.4f}", flush=True)


if __name__ == "__main__":
    main()
