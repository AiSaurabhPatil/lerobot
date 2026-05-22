#!/usr/bin/env python

"""Bridge an Isaac Sim USD scene to the LeRobot ROS Isaac robot adapter.

Run this with Isaac Sim's Python, not plain system Python.
"""

from __future__ import annotations

import argparse
import json
import pickle
import socket
import struct
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np


BRIDGE_VERSION = "tcp-threaded-camera-v2"
DEFAULT_USD_PATH = "/home/saurabh/Development/ARX_Model/AC one/acone_scene.usd"
DEFAULT_ROBOT_PRIM_PATH = "/World/ACone"

DEFAULT_POLICY_JOINTS = [
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

DEFAULT_GRIPPER_JOINTS = {
    "left_gripper": ["left_joint7", "left_joint8"],
    "right_gripper": ["right_joint17", "right_joint18"],
}

DEFAULT_CAMERAS = {
    "head_camera": "/World/ACone/base_link/head_camera",
    "left_wrist_camera": "/World/ACone/left_link6/left_wrist_camera",
    "right_wrist_camera": "/World/ACone/right_link16/right_wrist_camera",
}

OBS_IMAGE_PREFIX = "observation.images."
OBS_STATE = "observation.state"
ACTION = "action"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--usd-path", default=DEFAULT_USD_PATH)
    parser.add_argument("--robot-prim-path", default=DEFAULT_ROBOT_PRIM_PATH)
    parser.add_argument(
        "--dataset-root",
        default="",
        help="Optional LeRobot dataset/checkpoint dataset root. Reads meta/info.json for state/action/camera names.",
    )
    parser.add_argument(
        "--robot-config-json",
        default="",
        help=(
            "Optional JSON mapping for robots whose dataset names differ from Isaac DOF/camera names. "
            "Supports joint_map, action_joint_map, observation_joint_map, gripper_joints, camera_map."
        ),
    )
    parser.add_argument(
        "--robot-config",
        default="",
        help="Optional robot config file in YAML or JSON. Supports the quest3_streamer robot YAML schema.",
    )
    parser.add_argument("--joint-names", nargs="+", default=DEFAULT_POLICY_JOINTS)
    parser.add_argument("--action-joint-names", nargs="+", default=None)
    parser.add_argument("--observation-joint-names", nargs="+", default=None)
    parser.add_argument("--camera", action="append", default=[])
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--gripper-open-position", type=float, default=0.06)
    parser.add_argument("--gripper-closed-position", type=float, default=0.0)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--camera-warmup-frames", type=int, default=30)
    parser.add_argument("--action-stats-json", default="")
    parser.add_argument("--max-joint-target-step", type=float, default=0.05)
    parser.add_argument("--debug-jsonl", default="")
    parser.add_argument("--debug-every", type=int, default=1)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=0.0)
    return parser.parse_args()


def import_simulation_app():
    try:
        from isaacsim import SimulationApp
    except ImportError:
        from omni.isaac.kit import SimulationApp

    return SimulationApp


def import_isaac_runtime():
    try:
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.types import ArticulationAction
    except ImportError:
        from omni.isaac.core import World
        from omni.isaac.core.articulations import Articulation as SingleArticulation
        from omni.isaac.core.utils.types import ArticulationAction

    try:
        from isaacsim.sensors.camera import Camera
    except ImportError:
        from omni.isaac.sensor import Camera

    return World, SingleArticulation, Camera, ArticulationAction


def camera_map(items: list[str]) -> dict[str, str]:
    if not items:
        return dict(DEFAULT_CAMERAS)
    out = {}
    for item in items:
        name, prim_path = item.split("=", 1)
        out[name] = prim_path
    return out


def load_json_file(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def load_config_file(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        return load_json_file(path)
    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            f"PyYAML is required to read robot config {path}. "
            "Install pyyaml in the Isaac Python environment or provide a JSON config."
        ) from e
    with open(path) as f:
        return yaml.safe_load(f) or {}


def resolve_relative_path(path: str, config_path: str | Path | None) -> str:
    raw = Path(path).expanduser()
    if raw.is_absolute():
        return str(raw)
    candidates = [Path.cwd() / raw]
    if config_path is not None:
        cfg_parent = Path(config_path).resolve().parent
        candidates.extend([cfg_parent / raw, cfg_parent.parent / raw, cfg_parent.parent.parent / raw])
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[-1])


def load_dataset_info(dataset_root: str) -> dict[str, Any] | None:
    if not dataset_root:
        return None
    info_path = Path(dataset_root) / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Dataset info file not found: {info_path}")
    return load_json_file(info_path)


def feature_names(info: dict[str, Any] | None, feature_key: str) -> list[str] | None:
    if info is None:
        return None
    names = info.get("features", {}).get(feature_key, {}).get("names")
    return list(names) if names else None


def dataset_camera_names(info: dict[str, Any] | None) -> list[str]:
    if info is None:
        return []
    names = []
    for key, spec in info.get("features", {}).items():
        if key.startswith(OBS_IMAGE_PREFIX) and spec.get("dtype") in {"video", "image"}:
            names.append(key.removeprefix(OBS_IMAGE_PREFIX))
    return names


def load_robot_config(path: str) -> dict[str, Any]:
    if not path:
        return {}
    raw = load_config_file(path)
    return normalize_robot_config(raw, path)


def normalize_robot_config(raw: dict[str, Any], config_path: str | Path) -> dict[str, Any]:
    config = dict(raw)

    if "usd" in raw and "usd_path" not in config:
        config["usd_path"] = resolve_relative_path(str(raw["usd"]), config_path)

    if "prim_search_paths" in raw and "robot_prim_search_paths" not in config:
        config["robot_prim_search_paths"] = list(raw["prim_search_paths"])

    grippers = raw.get("grippers", {})
    gripper_joints = dict(config.get("gripper_joints", {}))
    if "left_joints" in grippers:
        gripper_joints["left_gripper"] = list(grippers["left_joints"])
    if "right_joints" in grippers:
        gripper_joints["right_gripper"] = list(grippers["right_joints"])
    if gripper_joints:
        config["gripper_joints"] = gripper_joints

    if "open_position" in grippers and "gripper_open_position" not in config:
        config["gripper_open_position"] = float(grippers["open_position"])
    if "closed_position" in grippers and "gripper_closed_position" not in config:
        config["gripper_closed_position"] = float(grippers["closed_position"])

    camera_map_cfg = dict(config.get("camera_map", {}))
    for camera_name, camera_spec in raw.get("cameras", {}).items():
        if isinstance(camera_spec, dict) and "prim_path" in camera_spec:
            camera_map_cfg[camera_name] = camera_spec["prim_path"]
    if camera_map_cfg:
        config["camera_map"] = camera_map_cfg

    if "joint_names" not in config:
        left_joints = raw.get("left_arm", {}).get("joints", [])
        right_joints = raw.get("right_arm", {}).get("joints", [])
        if left_joints or right_joints:
            config["joint_names"] = [*left_joints, "left_gripper", *right_joints, "right_gripper"]

    return config


def merged_joint_map(robot_config: dict[str, Any], kind: str) -> dict[str, Any]:
    joint_map = dict(robot_config.get("joint_map", {}))
    joint_map.update(robot_config.get(f"{kind}_joint_map", {}))
    return joint_map


def resolve_joint_names(
    args: argparse.Namespace,
    info: dict[str, Any] | None,
    robot_config: dict[str, Any],
) -> tuple[list[str], list[str]]:
    config_names = robot_config.get("joint_names", [])
    fallback_names = config_names or args.joint_names
    obs_names = args.observation_joint_names or feature_names(info, OBS_STATE) or fallback_names
    action_names = args.action_joint_names or feature_names(info, ACTION) or fallback_names
    return list(obs_names), list(action_names)


def resolve_camera_map(
    cli_items: list[str],
    robot_config: dict[str, Any],
    info: dict[str, Any] | None,
) -> dict[str, str]:
    if cli_items:
        return camera_map(cli_items)

    configured = dict(robot_config.get("camera_map", {}))
    camera_names = dataset_camera_names(info)
    if not camera_names:
        return configured or dict(DEFAULT_CAMERAS)

    resolved = {}
    for name in camera_names:
        if name in configured:
            resolved[name] = configured[name]
        elif name in DEFAULT_CAMERAS:
            resolved[name] = DEFAULT_CAMERAS[name]
        else:
            resolved[name] = ""
    return resolved


def available_camera_paths(stage) -> list[str]:
    return [str(prim.GetPath()) for prim in stage.Traverse() if prim.GetTypeName() == "Camera"]


def infer_camera_prim_path(stage, camera_name: str) -> str:
    candidates = available_camera_paths(stage)
    exact = [path for path in candidates if path.rsplit("/", 1)[-1] == camera_name]
    if len(exact) == 1:
        return exact[0]
    contains = [path for path in candidates if camera_name in path.rsplit("/", 1)[-1]]
    if len(contains) == 1:
        return contains[0]
    raise ValueError(
        f"Could not infer USD camera for dataset camera {camera_name!r}. "
        f"Pass --camera {camera_name}=/Usd/Camera/Path or set camera_map in --robot-config-json. "
        f"Available Camera prims: {candidates}"
    )


def resolve_camera_prim_path(stage, prim_path: str) -> str:
    prim = stage.GetPrimAtPath(prim_path)
    if prim and prim.IsValid():
        if prim.GetTypeName() == "Camera":
            return prim_path
        child_cameras = [
            str(child.GetPath())
            for child in stage.Traverse()
            if child.GetTypeName() == "Camera" and str(child.GetPath()).startswith(f"{prim_path}/")
        ]
        if len(child_cameras) == 1:
            return child_cameras[0]
        if child_cameras:
            raise ValueError(f"Camera path {prim_path!r} has multiple Camera children: {child_cameras}")

    cameras = available_camera_paths(stage)
    raise ValueError(f"Camera prim {prim_path!r} was not found as a Camera. Available Camera prims: {cameras}")


def resolve_robot_prim_path(stage, requested_path: str, robot_config: dict[str, Any]) -> str:
    if requested_path != DEFAULT_ROBOT_PRIM_PATH:
        return requested_path
    configured = robot_config.get("robot_prim_path")
    if configured:
        return str(configured)
    for prim_path in robot_config.get("robot_prim_search_paths", []):
        prim = stage.GetPrimAtPath(prim_path)
        if prim and prim.IsValid():
            return str(prim_path)
    return requested_path


def normalize_rgb_frame(frame: Any, width: int, height: int) -> np.ndarray | None:
    if frame is None:
        return None
    frame = np.asarray(frame)
    if frame.size == 0:
        return None
    if frame.ndim == 1:
        if frame.size == width * height * 4:
            frame = frame.reshape(height, width, 4)
        elif frame.size == width * height * 3:
            frame = frame.reshape(height, width, 3)
        else:
            raise ValueError(f"Unexpected flat camera buffer size: {frame.size}")
    if frame.ndim != 3 or frame.shape[2] < 3:
        raise ValueError(f"Unexpected camera frame shape: {frame.shape}")
    rgb = frame[:, :, :3]
    if rgb.dtype != np.uint8:
        max_value = float(np.nanmax(rgb)) if rgb.size else 0.0
        if max_value <= 1.0:
            rgb = rgb * 255.0
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(rgb)


def camera_rgb_frame(camera, width: int, height: int) -> np.ndarray | None:
    for frame in (
        getattr(camera, "get_rgb", lambda: None)(),
        getattr(camera, "get_rgba", lambda: None)(),
        camera.get_current_frame().get("rgb") if hasattr(camera, "get_current_frame") else None,
    ):
        rgb = normalize_rgb_frame(frame, width, height)
        if rgb is not None:
            return rgb
    return None


def step_replicator_once(rep: Any | None) -> None:
    if rep is None:
        return
    try:
        rep.orchestrator.step(delta_time=0.0, pause_timeline=False, wait_for_render=True)
    except TypeError:
        rep.orchestrator.step()


def send_msg(sock: socket.socket, payload: dict[str, Any]) -> None:
    data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack("!I", len(data)) + data)


def recv_msg(sock: socket.socket) -> dict[str, Any]:
    header = recv_exact(sock, 4)
    size = struct.unpack("!I", header)[0]
    return pickle.loads(recv_exact(sock, size))


def recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = sock.recv(size - len(chunks))
        if not chunk:
            raise ConnectionError("socket closed")
        chunks.extend(chunk)
    return bytes(chunks)


def joint_indices(dof_names: list[str], names: list[str]) -> np.ndarray:
    return np.array([dof_names.index(name) for name in names], dtype=np.int32)


def clean_action_name(name: str) -> str:
    return name.removesuffix(".pos")


def mapped_dof_names(name: str, joint_map: dict[str, Any], gripper_joints: dict[str, list[str]]) -> list[str]:
    name = clean_action_name(name)
    mapped = joint_map.get(name, name)
    if isinstance(mapped, str):
        if mapped in gripper_joints:
            return list(gripper_joints[mapped])
        return [mapped]
    return list(mapped)


def gripper_to_scalar(
    positions: np.ndarray,
    dof_names: list[str],
    gripper_joints: list[str],
    closed: float,
    open_: float,
) -> float:
    indices = joint_indices(dof_names, gripper_joints)
    value = float(np.mean(positions[indices]))
    if open_ == closed:
        return value
    return float(np.clip((value - open_) / (closed - open_), 0.0, 1.0))


def read_policy_position(
    name: str,
    positions: np.ndarray,
    dof_names: list[str],
    joint_map: dict[str, Any],
    gripper_joints: dict[str, list[str]],
    closed: float,
    open_: float,
) -> float:
    mapped = mapped_dof_names(name, joint_map, gripper_joints)
    raw_name = clean_action_name(name)
    if any(dof_name not in dof_names for dof_name in mapped) and raw_name in dof_names:
        mapped = [raw_name]
    if len(mapped) > 1:
        return gripper_to_scalar(positions, dof_names, mapped, closed, open_)
    dof_name = mapped[0]
    if dof_name not in dof_names:
        raise ValueError(f"Dataset joint {name!r} maps to missing Isaac DOF {dof_name!r}")
    return float(positions[dof_names.index(dof_name)])


def expand_policy_targets(
    policy_targets: dict[str, float],
    dof_names: list[str],
    joint_map: dict[str, Any],
    gripper_joints: dict[str, list[str]],
    closed: float,
    open_: float,
) -> tuple[np.ndarray, np.ndarray]:
    names: list[str] = []
    values: list[float] = []
    for name, value in policy_targets.items():
        action_name = clean_action_name(name)
        mapped = mapped_dof_names(action_name, joint_map, gripper_joints)
        if any(dof_name not in dof_names for dof_name in mapped) and action_name in dof_names:
            mapped = [action_name]
        if len(mapped) > 1:
            missing = [gripper_joint for gripper_joint in mapped if gripper_joint not in dof_names]
            if missing:
                raise ValueError(f"Dataset action {action_name!r} maps to missing Isaac DOFs: {missing}")
            physical_value = open_ + float(np.clip(value, 0.0, 1.0)) * (closed - open_)
            for gripper_joint in mapped:
                names.append(gripper_joint)
                values.append(physical_value)
        elif mapped[0] in dof_names:
            names.append(mapped[0])
            values.append(float(value))

    if not names:
        raise ValueError(f"No Isaac DOF targets matched policy action keys: {sorted(policy_targets)}")

    return joint_indices(dof_names, names), np.array(values, dtype=np.float32)


def load_action_limits(path: str, joint_names: list[str]) -> tuple[dict[str, float], dict[str, float]] | None:
    if not path:
        return None
    with open(path) as f:
        stats = json.load(f)["action"]
    return (
        {name: float(value) for name, value in zip(joint_names, stats["min"], strict=False)},
        {name: float(value) for name, value in zip(joint_names, stats["max"], strict=False)},
    )


def clip_policy_targets(
    policy_targets: dict[str, float],
    action_limits: tuple[dict[str, float], dict[str, float]] | None,
) -> dict[str, float]:
    if action_limits is None:
        return policy_targets
    mins, maxs = action_limits
    clipped = {}
    for key, value in policy_targets.items():
        name = key.removesuffix(".pos")
        clipped[key] = float(np.clip(value, mins[name], maxs[name])) if name in mins and name in maxs else value
    return clipped


def limit_target_step(
    target_indices: np.ndarray,
    target_values: np.ndarray,
    current_positions: np.ndarray,
    max_step: float,
) -> np.ndarray:
    if max_step <= 0:
        return target_values
    current = current_positions[target_indices]
    return current + np.clip(target_values - current, -max_step, max_step)


def write_debug_jsonl(path: str, every: int, counter: int, payload: dict[str, Any]) -> None:
    if not path or counter % max(1, every) != 0:
        return
    record = {"step": counter, "time": time.time()}
    for key, value in payload.items():
        if isinstance(value, np.ndarray):
            record[key] = value.tolist()
        else:
            record[key] = value
    line = json.dumps(record)
    if path in {"-", "stdout", "terminal"}:
        print(f"ISAAC_BRIDGE_DEBUG {line}", flush=True)
        return
    with open(path, "a") as f:
        f.write(line + "\n")


def apply_joint_position_targets(
    robot: Any,
    controller: Any,
    articulation_action_cls: type,
    target_indices: np.ndarray,
    target_values: np.ndarray,
) -> None:
    if hasattr(controller, "set_joint_position_targets"):
        controller.set_joint_position_targets(target_values, joint_indices=target_indices)
        return

    action = articulation_action_cls(joint_positions=target_values, joint_indices=target_indices)
    if hasattr(controller, "apply_action"):
        controller.apply_action(action)
    else:
        robot.apply_action(action)


def serve_policy_client(
    host: str,
    port: int,
    shared: dict[str, Any],
    condition: threading.Condition,
    stop_event: threading.Event,
) -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(1)
    server.settimeout(0.5)
    print(f"Waiting for LeRobot policy client on {host}:{port}")
    try:
        while not stop_event.is_set():
            try:
                client, addr = server.accept()
            except socket.timeout:
                continue

            print(f"LeRobot policy client connected from {addr}")
            with client:
                client.settimeout(5.0)
                while not stop_event.is_set():
                    try:
                        request = recv_msg(client)
                        msg_type = request.get("type")
                        if msg_type == "hello":
                            send_msg(client, {"type": "hello_ack"})
                        elif msg_type == "get_observation":
                            with condition:
                                condition.wait_for(
                                    lambda: shared.get("latest_observation") is not None
                                    or stop_event.is_set(),
                                    timeout=5.0,
                                )
                                observation = shared.get("latest_observation")
                            if observation is None:
                                send_msg(client, {"type": "error", "message": "No observation available"})
                            else:
                                send_msg(client, {"type": "observation", "observation": observation})
                        elif msg_type == "action":
                            with condition:
                                shared["latest_action"] = request["action"]
                            send_msg(client, {"type": "action_ack", "action": request["action"]})
                        elif msg_type == "close":
                            break
                    except (ConnectionError, socket.timeout):
                        break
                    except Exception:
                        traceback.print_exc()
                        break
            print(f"Waiting for LeRobot policy client on {host}:{port}")
    finally:
        server.close()


def main() -> None:
    args = parse_args()
    dataset_info = load_dataset_info(args.dataset_root)
    robot_config = load_robot_config(args.robot_config or args.robot_config_json)
    if args.usd_path == DEFAULT_USD_PATH and robot_config.get("usd_path"):
        args.usd_path = str(robot_config["usd_path"])
    observation_joint_names, action_joint_names = resolve_joint_names(args, dataset_info, robot_config)
    gripper_joints = {
        name: list(joints) for name, joints in robot_config.get("gripper_joints", DEFAULT_GRIPPER_JOINTS).items()
    }
    observation_joint_map = merged_joint_map(robot_config, "observation")
    action_joint_map = merged_joint_map(robot_config, "action")
    camera_config = resolve_camera_map(args.camera, robot_config, dataset_info)
    if "gripper_open_position" in robot_config and args.gripper_open_position == 0.06:
        args.gripper_open_position = float(robot_config["gripper_open_position"])
    if "gripper_closed_position" in robot_config and args.gripper_closed_position == 0.0:
        args.gripper_closed_position = float(robot_config["gripper_closed_position"])
    if dataset_info is not None and args.fps == 30.0:
        args.fps = float(dataset_info.get("fps", args.fps))

    print(f"LeRobot Isaac bridge version: {BRIDGE_VERSION}", flush=True)
    print(f"Bridge script path: {__file__}", flush=True)
    print(f"USD path: {args.usd_path}", flush=True)
    print(f"Observation joints ({len(observation_joint_names)}): {observation_joint_names}", flush=True)
    print(f"Action joints ({len(action_joint_names)}): {action_joint_names}", flush=True)
    SimulationApp = import_simulation_app()
    simulation_app = SimulationApp({"headless": args.headless})

    import omni.usd

    try:
        import omni.replicator.core as rep
    except Exception:
        rep = None

    World, SingleArticulation, Camera, ArticulationAction = import_isaac_runtime()

    usd_context = omni.usd.get_context()
    usd_context.open_stage(args.usd_path)
    while usd_context.get_stage() is None:
        simulation_app.update()

    stage = usd_context.get_stage()
    args.robot_prim_path = resolve_robot_prim_path(stage, args.robot_prim_path, robot_config)
    print(f"Robot prim path: {args.robot_prim_path}", flush=True)

    world = World(stage_units_in_meters=1.0)
    world.reset()

    robot = SingleArticulation(prim_path=args.robot_prim_path, name="lerobot_robot")
    robot.initialize()
    controller = robot.get_articulation_controller()

    cameras = {}
    if dataset_info is not None and not camera_config and dataset_camera_names(dataset_info):
        raise ValueError(
            "Dataset has image features, but no camera mapping was resolved. "
            "Pass --camera name=/Usd/Camera/Path or set camera_map in --robot-config-json."
        )
    for name, prim_path in camera_config.items():
        if not prim_path:
            prim_path = infer_camera_prim_path(stage, name)
        camera_prim_path = resolve_camera_prim_path(stage, prim_path)
        if camera_prim_path != prim_path:
            print(f"Resolved camera {name!r}: {prim_path} -> {camera_prim_path}", flush=True)
        cam = Camera(prim_path=camera_prim_path, name=name, resolution=(args.width, args.height))
        cam.initialize()
        if hasattr(cam, "resume"):
            cam.resume()
        cameras[name] = cam
    last_images = {
        name: np.zeros((args.height, args.width, 3), dtype=np.uint8) for name in cameras
    }
    camera_failures = {name: 0 for name in cameras}

    if cameras and args.camera_warmup_frames > 0:
        print(f"Warming {len(cameras)} Isaac camera render products...", flush=True)
        pending = set(cameras)
        for _ in range(args.camera_warmup_frames):
            if not pending:
                break
            world.step(render=True)
            for name in list(pending):
                image = camera_rgb_frame(cameras[name], args.width, args.height)
                if image is None:
                    step_replicator_once(rep)
                    image = camera_rgb_frame(cameras[name], args.width, args.height)
                if image is not None:
                    last_images[name] = image
                    pending.remove(name)
        if pending:
            print(
                "Warning: no RGB frame yet from Isaac camera(s): "
                f"{sorted(pending)}. Sending black frames until Isaac produces data.",
                flush=True,
            )

    joint_targets: tuple[np.ndarray, np.ndarray] | None = None

    dof_names = list(getattr(robot, "dof_names", []) or [])
    if not dof_names:
        raise RuntimeError("Could not read articulation dof_names from Isaac robot.")
    stats_json = args.action_stats_json
    if not stats_json and args.dataset_root:
        candidate = Path(args.dataset_root) / "meta" / "stats.json"
        if candidate.is_file():
            stats_json = str(candidate)
    action_limits = load_action_limits(stats_json, action_joint_names)

    shared: dict[str, Any] = {"latest_observation": None, "latest_action": None}
    condition = threading.Condition()
    stop_event = threading.Event()
    server_thread = threading.Thread(
        target=serve_policy_client,
        args=(args.host, args.port, shared, condition, stop_event),
        daemon=True,
    )
    server_thread.start()

    dt = 1.0 / args.fps
    start_time = time.perf_counter()
    debug_counter = 0
    try:
        while not stop_event.is_set():
            if args.max_seconds > 0 and (time.perf_counter() - start_time) >= args.max_seconds:
                print(f"Max runtime reached ({args.max_seconds}s); shutting down.")
                break

            start = time.perf_counter()
            with condition:
                action = shared.get("latest_action")
                shared["latest_action"] = None

            if action is not None:
                try:
                    action = clip_policy_targets(action, action_limits)
                    joint_targets = expand_policy_targets(
                        action,
                        dof_names,
                        action_joint_map,
                        gripper_joints,
                        args.gripper_closed_position,
                        args.gripper_open_position,
                    )
                except Exception:
                    traceback.print_exc()

            if joint_targets is not None:
                try:
                    target_indices, target_values = joint_targets
                    raw_target_values = target_values.copy()
                    current_positions = np.asarray(robot.get_joint_positions(), dtype=np.float32)
                    target_values = limit_target_step(
                        target_indices,
                        target_values,
                        current_positions,
                        args.max_joint_target_step,
                    )
                    debug_counter += 1
                    write_debug_jsonl(
                        args.debug_jsonl,
                        args.debug_every,
                        debug_counter,
                        {
                            "target_joint_indices": target_indices,
                            "current_positions": current_positions[target_indices],
                            "raw_target_values": raw_target_values,
                            "applied_target_values": target_values,
                        },
                    )
                    apply_joint_position_targets(
                        robot,
                        controller,
                        ArticulationAction,
                        target_indices,
                        target_values,
                    )
                except Exception:
                    traceback.print_exc()
                    joint_targets = None

            try:
                world.step(render=True)
            except Exception:
                print("Isaac world.step(render=True) failed:")
                traceback.print_exc()
                time.sleep(0.1)
                continue

            try:
                positions = np.asarray(robot.get_joint_positions(), dtype=np.float32)
            except Exception:
                print("Could not read Isaac robot joint positions:")
                traceback.print_exc()
                time.sleep(0.1)
                continue

            policy_positions = []
            for name in observation_joint_names:
                policy_positions.append(
                    read_policy_position(
                        name,
                        positions,
                        dof_names,
                        observation_joint_map,
                        gripper_joints,
                        args.gripper_closed_position,
                        args.gripper_open_position,
                    )
                )

            observation = {
                f"{name}.pos": float(value)
                for name, value in zip(observation_joint_names, policy_positions, strict=False)
            }
            for name, cam in cameras.items():
                try:
                    image = camera_rgb_frame(cam, args.width, args.height)
                    if image is None:
                        step_replicator_once(rep)
                        image = camera_rgb_frame(cam, args.width, args.height)
                    if image is not None:
                        last_images[name] = image
                        camera_failures[name] = 0
                    else:
                        camera_failures[name] += 1
                        if camera_failures[name] in (1, args.camera_warmup_frames):
                            print(f"Isaac camera {name!r} has not produced an RGB frame yet.", flush=True)
                except Exception:
                    camera_failures[name] += 1
                    if camera_failures[name] in (1, args.camera_warmup_frames) or camera_failures[name] % 300 == 0:
                        print(f"Could not read Isaac camera '{name}':")
                        traceback.print_exc()
                observation[name] = last_images[name]

            with condition:
                shared["latest_observation"] = observation
                condition.notify_all()

            sleep_s = dt - (time.perf_counter() - start)
            if sleep_s > 0:
                time.sleep(sleep_s)
    except KeyboardInterrupt:
        print("Interrupted by user; shutting down.")
    finally:
        stop_event.set()
        with condition:
            condition.notify_all()
        server_thread.join(timeout=1.0)
        simulation_app.close()


if __name__ == "__main__":
    main()
