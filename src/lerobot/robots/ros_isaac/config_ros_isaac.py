#!/usr/bin/env python

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lerobot.robots.config import RobotConfig


DEFAULT_PI05_JOINTS = [
    "left_joint1",
    "left_joint2",
    "left_joint3",
    "left_joint4",
    "left_joint5",
    "left_joint6",
    "left_gripper",
    "right_joint11",
    "right_joint12",
    "right_joint13",
    "right_joint14",
    "right_joint15",
    "right_joint16",
    "right_gripper",
]


@RobotConfig.register_subclass("ros_isaac")
@dataclass(kw_only=True)
class RosIsaacRobotConfig(RobotConfig):
    robot_config: str | None = None
    joint_names: list[str] = field(default_factory=lambda: list(DEFAULT_PI05_JOINTS))
    joint_state_topic: str = "/joint_states"
    action_topic: str = "/lerobot/joint_targets"
    camera_topics: dict[str, str] = field(
        default_factory=lambda: {
            "head_camera": "/camera/head_camera/image_raw",
            "left_wrist_camera": "/camera/left_wrist_camera/image_raw",
            "right_wrist_camera": "/camera/right_wrist_camera/image_raw",
        }
    )
    camera_shapes: dict[str, tuple[int, int, int]] = field(
        default_factory=lambda: {
            "head_camera": (224, 224, 3),
            "left_wrist_camera": (224, 224, 3),
            "right_wrist_camera": (224, 224, 3),
        }
    )
    ros_node_name: str = "lerobot_ros_isaac_robot"
    bridge_host: str = "127.0.0.1"
    bridge_port: int = 8765
    observation_timeout_s: float = 5.0
    action_duration_s: float = 0.05
    publish_velocities: bool = False

    def __post_init__(self):
        if self.robot_config:
            cfg = _load_robot_config(self.robot_config)
            joint_names = _joint_names_from_config(cfg)
            if joint_names:
                self.joint_names = joint_names

            camera_topics = _camera_topics_from_config(cfg)
            if camera_topics:
                self.camera_topics = camera_topics
                self.camera_shapes = {
                    name: self.camera_shapes.get(name, (224, 224, 3)) for name in camera_topics
                }

        super().__post_init__()


def _load_robot_config(path: str) -> dict[str, Any]:
    path_obj = Path(path)
    if path_obj.suffix.lower() == ".json":
        import json

        with open(path_obj) as f:
            return json.load(f)

    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            f"PyYAML is required to read robot config {path_obj}. "
            "Install pyyaml in the rollout environment or provide a JSON config."
        ) from e

    with open(path_obj) as f:
        return yaml.safe_load(f) or {}


def _joint_names_from_config(cfg: dict[str, Any]) -> list[str]:
    if "joint_names" in cfg:
        return list(cfg["joint_names"])

    left_joints = list(cfg.get("left_arm", {}).get("joints", []))
    right_joints = list(cfg.get("right_arm", {}).get("joints", []))
    if not left_joints and not right_joints:
        return []

    names = []
    names.extend(left_joints)
    if cfg.get("grippers", {}).get("left_joints"):
        names.append("left_gripper")
    names.extend(right_joints)
    if cfg.get("grippers", {}).get("right_joints"):
        names.append("right_gripper")
    return names


def _camera_topics_from_config(cfg: dict[str, Any]) -> dict[str, str]:
    topics = {}
    for name, camera_cfg in cfg.get("cameras", {}).items():
        if isinstance(camera_cfg, dict):
            topics[name] = camera_cfg.get("topic", f"/camera/{name}/image_raw")
    return topics
