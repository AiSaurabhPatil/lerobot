#!/usr/bin/env python

import logging
import pickle
import socket
import struct
from functools import cached_property
from typing import Any

from lerobot.types import RobotAction, RobotObservation

from ..robot import Robot
from .config_ros_isaac import RosIsaacRobotConfig

logger = logging.getLogger(__name__)


class RosIsaacRobot(Robot):
    config_class = RosIsaacRobotConfig
    name = "ros_isaac"

    def __init__(self, config: RosIsaacRobotConfig):
        super().__init__(config)
        self.config = config
        self._socket: socket.socket | None = None

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {
            **{f"{joint}.pos": float for joint in self.config.joint_names},
            **dict(self.config.camera_shapes),
        }

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {f"{joint}.pos": float for joint in self.config.joint_names}

    @property
    def is_connected(self) -> bool:
        return self._socket is not None

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise RuntimeError(f"{self} is already connected")
        sock = socket.create_connection(
            (self.config.bridge_host, self.config.bridge_port),
            timeout=self.config.observation_timeout_s,
        )
        sock.settimeout(self.config.observation_timeout_s)
        self._socket = sock
        self._send({"type": "hello", "role": "policy"})
        reply = self._recv()
        if reply.get("type") != "hello_ack":
            raise RuntimeError(f"Unexpected Isaac bridge reply: {reply.get('type')}")
        logger.info(
            "%s connected to Isaac bridge at %s:%d",
            self,
            self.config.bridge_host,
            self.config.bridge_port,
        )

    def _send(self, payload: dict[str, Any]) -> None:
        if self._socket is None:
            raise RuntimeError(f"{self} is not connected")
        data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        self._socket.sendall(struct.pack("!I", len(data)) + data)

    def _recv(self) -> dict[str, Any]:
        if self._socket is None:
            raise RuntimeError(f"{self} is not connected")
        header = self._recv_exact(4)
        size = struct.unpack("!I", header)[0]
        return pickle.loads(self._recv_exact(size))

    def _recv_exact(self, size: int) -> bytes:
        if self._socket is None:
            raise RuntimeError(f"{self} is not connected")
        chunks = bytearray()
        while len(chunks) < size:
            chunk = self._socket.recv(size - len(chunks))
            if not chunk:
                raise ConnectionError("Isaac bridge socket closed")
            chunks.extend(chunk)
        return bytes(chunks)

    def get_observation(self) -> RobotObservation:
        self._send({"type": "get_observation"})
        reply = self._recv()
        if reply.get("type") != "observation":
            raise RuntimeError(f"Unexpected Isaac bridge reply: {reply.get('type')}")
        return reply["observation"]

    def send_action(self, action: RobotAction) -> RobotAction:
        self._send({"type": "action", "action": dict(action)})
        reply = self._recv()
        if reply.get("type") != "action_ack":
            raise RuntimeError(f"Unexpected Isaac bridge reply: {reply.get('type')}")
        return reply["action"]

    def disconnect(self) -> None:
        if self._socket is None:
            return
        try:
            self._send({"type": "close"})
        except Exception:
            pass
        self._socket.close()
        self._socket = None
        logger.info("%s disconnected.", self)
