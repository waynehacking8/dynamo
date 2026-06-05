# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared Dynamo snapshot helpers for checkpoint lifecycle."""

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

from dynamo.common.utils.namespace import get_worker_namespace

logger = logging.getLogger(__name__)
PODINFO_ROOT = "/etc/podinfo"
KUBERNETES_REQUIRED_PODINFO_FILES = {
    "DYN_NAMESPACE": "dyn_namespace",
    "DYN_COMPONENT": "dyn_component",
    "DYN_PARENT_DGD_K8S_NAME": "dyn_parent_dgd_k8s_name",
    "DYN_PARENT_DGD_K8S_NAMESPACE": "dyn_parent_dgd_k8s_namespace",
}
KUBERNETES_OPTIONAL_PODINFO_FILES = {
    "DYN_NAMESPACE_WORKER_SUFFIX": "dyn_namespace_worker_suffix",
}
RESTORE_RUNTIME_ENV_PODINFO_FILE = "dyn_restore_runtime_env"
EngineT = TypeVar("EngineT")

# Must match snapshotprotocol.{SnapshotCompleteFile,RestoreCompleteFile,ReadyForCheckpointFile}.
SNAPSHOT_COMPLETE_FILE = "snapshot-complete"
RESTORE_COMPLETE_FILE = "restore-complete"
READY_FOR_CHECKPOINT_FILE = "ready-for-checkpoint"

# Poll interval for the snapshot-control directory. Checkpoint and restore
# latencies are seconds, so 100ms is negligible overhead.
_SENTINEL_POLL_INTERVAL_SEC = 0.1


RESTORE_RUNTIME_ENV_NAMES = {
    # Parsed Python runtime config that must also refresh the in-memory config
    # passed to create_runtime().
    "DYN_DISCOVERY_BACKEND",
    "DYN_REQUEST_PLANE",
    "DYN_EVENT_PLANE",
    # DistributedRuntime infrastructure env read after restore.
    "NATS_SERVER",
    "ETCD_ENDPOINTS",
    # Runtime system server/readiness env read after restore.
    "DYN_SYSTEM_PORT",
    "DYN_HEALTH_CHECK_ENABLED",
    "DYN_SYSTEM_STARTING_HEALTH_STATUS",
    "DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS",
    "DYN_SYSTEM_HOST",
    "DYN_SYSTEM_HEALTH_PATH",
    "DYN_SYSTEM_LIVE_PATH",
    # Kubernetes discovery mode env read when the restored runtime registers.
    "DYN_KUBE_DISCOVERY_MODE",
    "CONTAINER_NAME",
}
RESTORE_RUNTIME_ENV_DEFAULTS_WHEN_UNSET = {
    "DYN_DISCOVERY_BACKEND": "etcd",
    "DYN_REQUEST_PLANE": "tcp",
    "DYN_EVENT_PLANE": None,
}


@dataclass(frozen=True)
class RestoreRuntimeConfig:
    """Runtime config refreshed from the restore pod after CRIU restore.

    CRIU restores the checkpoint-time process environment. Snapshot restore
    reads non-secret restore-time config from /etc/podinfo and applies it before
    creating ``DistributedRuntime``.

    Attributes:
        namespace: Dynamo worker namespace after applying any worker suffix.
        discovery_backend: Restore-time discovery backend.
        request_plane: Restore-time request plane, when explicitly configured.
        event_plane: Restore-time event plane, when explicitly configured.
    """

    namespace: str
    discovery_backend: str
    request_plane: str | None = None
    event_plane: str | None = None


class CheckpointConfig:
    """Parsed checkpoint configuration plus the sentinel-driven lifecycle."""

    def __init__(self, control_dir: str):
        self.control_dir = control_dir
        self.ready_file = os.path.join(control_dir, READY_FOR_CHECKPOINT_FILE)

    @classmethod
    def from_env(cls) -> "CheckpointConfig | None":
        control_dir = os.environ.get("DYN_SNAPSHOT_CONTROL_DIR")
        if not control_dir:
            return None

        configure_checkpoint_transport_env()
        return cls(control_dir=control_dir)

    async def run_lifecycle(
        self,
        pause_controller: Any,
        *pause_args: object,
    ) -> bool:
        logger.info("Pausing model")
        await pause_controller.pause(*pause_args)

        try:
            with open(self.ready_file, "w", encoding="utf-8") as ready_file:
                ready_file.write("ready")

            logger.info(
                "Ready for checkpoint. Polling for sentinel in %s "
                "(snapshot-complete or restore-complete)",
                self.control_dir,
            )

            event = await self._wait_for_sentinel()
        finally:
            self._cleanup_ready_and_sentinels()

        if event == "restore":
            logger.info("Restore sentinel detected")
            logger.info("Resuming model after restore")
            await pause_controller.resume()
            pause_controller.mark_resumed()
            return True

        logger.info("Snapshot completion sentinel detected")
        return False

    async def _wait_for_sentinel(self) -> str:
        snapshot_path = Path(self.control_dir) / SNAPSHOT_COMPLETE_FILE
        restore_path = Path(self.control_dir) / RESTORE_COMPLETE_FILE
        while True:
            if snapshot_path.exists():
                return "checkpoint"
            if restore_path.exists():
                return "restore"
            await asyncio.sleep(_SENTINEL_POLL_INTERVAL_SEC)

    def _cleanup_ready_and_sentinels(self) -> None:
        for name in (
            READY_FOR_CHECKPOINT_FILE,
            SNAPSHOT_COMPLETE_FILE,
            RESTORE_COMPLETE_FILE,
        ):
            path = os.path.join(self.control_dir, name)
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
            except OSError:
                logger.exception("Failed to clean up %s at %s", name, path)


def configure_checkpoint_transport_env() -> None:
    gloo_ifname = os.environ.get("GLOO_SOCKET_IFNAME")
    if gloo_ifname and gloo_ifname != "lo":
        logger.warning(
            "Overriding GLOO_SOCKET_IFNAME=%r with 'lo' for checkpoint mode "
            "because CRIU cannot restore sockets bound to non-loopback addresses",
            gloo_ifname,
        )
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"

    nccl_ifname = os.environ.get("NCCL_SOCKET_IFNAME")
    if nccl_ifname and nccl_ifname != "lo":
        logger.warning(
            "Overriding NCCL_SOCKET_IFNAME=%r with 'lo' for checkpoint mode "
            "because CRIU cannot restore sockets bound to non-loopback addresses",
            nccl_ifname,
        )
    os.environ["NCCL_SOCKET_IFNAME"] = "lo"

    nccl_cumem_enable = os.environ.get("NCCL_CUMEM_ENABLE")
    if nccl_cumem_enable and nccl_cumem_enable != "0":
        logger.warning(
            "Overriding NCCL_CUMEM_ENABLE=%r with '0' for checkpoint mode "
            "because cuda-checkpoint does not support cuMem-backed NCCL allocations",
            nccl_cumem_enable,
        )
    os.environ["NCCL_CUMEM_ENABLE"] = "0"

    nccl_p2p_disable = os.environ.get("NCCL_P2P_DISABLE")
    if nccl_p2p_disable and nccl_p2p_disable != "0":
        logger.warning(
            "Overriding NCCL_P2P_DISABLE=%r with '0' for checkpoint mode "
            "to keep NCCL on GPU P2P transport when topology allows it",
            nccl_p2p_disable,
        )
    os.environ["NCCL_P2P_DISABLE"] = "0"

    nccl_nvls_enable = os.environ.get("NCCL_NVLS_ENABLE")
    if nccl_nvls_enable and nccl_nvls_enable != "0":
        logger.warning(
            "Overriding NCCL_NVLS_ENABLE=%r with '0' for checkpoint mode "
            "to avoid NVLS and keep NCCL on the legacy P2P path",
            nccl_nvls_enable,
        )
    os.environ["NCCL_NVLS_ENABLE"] = "0"

    nccl_ib_disable = os.environ.get("NCCL_IB_DISABLE")
    if nccl_ib_disable and nccl_ib_disable != "1":
        logger.warning(
            "Overriding NCCL_IB_DISABLE=%r with '1' for checkpoint mode "
            "because CRIU and cuda-checkpoint cannot restore InfiniBand state",
            nccl_ib_disable,
        )
    os.environ["NCCL_IB_DISABLE"] = "1"

    nccl_ras_enable = os.environ.get("NCCL_RAS_ENABLE")
    if nccl_ras_enable and nccl_ras_enable != "0":
        logger.warning(
            "Overriding NCCL_RAS_ENABLE=%r with '0' for checkpoint mode "
            "because NCCL RAS background state is not part of the checkpoint contract",
            nccl_ras_enable,
        )
    os.environ["NCCL_RAS_ENABLE"] = "0"

    torch_nccl_monitoring = os.environ.get("TORCH_NCCL_ENABLE_MONITORING")
    if torch_nccl_monitoring and torch_nccl_monitoring != "0":
        logger.warning(
            "Overriding TORCH_NCCL_ENABLE_MONITORING=%r with '0' for checkpoint mode "
            "because ProcessGroupNCCL monitoring can terminate restored processes",
            torch_nccl_monitoring,
        )
    os.environ["TORCH_NCCL_ENABLE_MONITORING"] = "0"
    os.environ.setdefault("TORCH_NCCL_DUMP_ON_TIMEOUT", "0")


@dataclass
class EngineSnapshotController(Generic[EngineT]):
    engine: EngineT
    pause_controller: Any
    checkpoint_config: CheckpointConfig
    pause_args: tuple[object, ...] = ()

    async def wait_for_restore(self) -> bool:
        return await self.checkpoint_config.run_lifecycle(
            self.pause_controller,
            *self.pause_args,
        )

    def reload_restore_identity(
        self,
        namespace: str,
        discovery_backend: str,
    ) -> tuple[str, str]:
        restored = reload_snapshot_restore_config(
            namespace=namespace,
            discovery_backend=discovery_backend,
        )
        return restored.namespace, restored.discovery_backend

    def reload_restore_config(
        self,
        namespace: str,
        discovery_backend: str,
        request_plane: str | None = None,
        event_plane: str | None = None,
    ) -> RestoreRuntimeConfig:
        return reload_snapshot_restore_config(
            namespace=namespace,
            discovery_backend=discovery_backend,
            request_plane=request_plane,
            event_plane=event_plane,
        )


def reload_snapshot_restore_identity(
    namespace: str,
    discovery_backend: str,
) -> tuple[str, str]:
    restored = reload_snapshot_restore_config(
        namespace=namespace,
        discovery_backend=discovery_backend,
    )
    return restored.namespace, restored.discovery_backend


def reload_snapshot_restore_config(
    namespace: str,
    discovery_backend: str,
    request_plane: str | None = None,
    event_plane: str | None = None,
) -> RestoreRuntimeConfig:
    """Reload restore-time Dynamo runtime env from the Downward API.

    The operator projects non-secret restore runtime settings into
    ``/etc/podinfo/dyn_restore_runtime_env``. Apply them before constructing
    ``DistributedRuntime`` so restored workers do not use stale checkpoint-job
    env such as ``NATS_SERVER=localhost`` or a missing ``DYN_SYSTEM_PORT``.
    """

    restore_env = _apply_restore_runtime_env_from_podinfo()

    refreshed_discovery_backend = _restore_env_value(
        restore_env,
        env_name="DYN_DISCOVERY_BACKEND",
        fallback=discovery_backend,
    )
    if refreshed_discovery_backend != "kubernetes":
        logger.info(
            "Snapshot restore reusing configured discovery backend",
            extra={
                "dynamo_namespace": namespace,
                "discovery_backend": refreshed_discovery_backend,
            },
        )
        return RestoreRuntimeConfig(
            namespace=namespace,
            discovery_backend=refreshed_discovery_backend,
            request_plane=_restore_env_value(
                restore_env,
                env_name="DYN_REQUEST_PLANE",
                fallback=request_plane,
            ),
            event_plane=_restore_env_value(
                restore_env,
                env_name="DYN_EVENT_PLANE",
                fallback=event_plane,
            ),
        )

    for env_name, podinfo_file in KUBERNETES_REQUIRED_PODINFO_FILES.items():
        podinfo_path = os.path.join(PODINFO_ROOT, podinfo_file)
        if not os.path.isfile(podinfo_path):
            raise RuntimeError(f"snapshot restore requires {podinfo_path}")

        with open(podinfo_path, encoding="utf-8") as podinfo:
            value = podinfo.read().strip()
        if not value:
            raise RuntimeError(f"snapshot restore requires a non-empty {podinfo_path}")

        os.environ[env_name] = value

    for env_name, podinfo_file in KUBERNETES_OPTIONAL_PODINFO_FILES.items():
        podinfo_path = os.path.join(PODINFO_ROOT, podinfo_file)
        if not os.path.isfile(podinfo_path):
            os.environ.pop(env_name, None)
            continue

        with open(podinfo_path, encoding="utf-8") as podinfo:
            value = podinfo.read().strip()
        if not value:
            os.environ.pop(env_name, None)
            continue

        os.environ[env_name] = value

    os.environ["DYN_DISCOVERY_BACKEND"] = "kubernetes"
    return RestoreRuntimeConfig(
        namespace=get_worker_namespace(),
        discovery_backend="kubernetes",
        request_plane=_restore_env_value(
            restore_env,
            env_name="DYN_REQUEST_PLANE",
            fallback=request_plane,
        ),
        event_plane=_restore_env_value(
            restore_env,
            env_name="DYN_EVENT_PLANE",
            fallback=event_plane,
        ),
    )


def _restore_env_value(
    restore_env: dict[str, str | None],
    env_name: str,
    fallback: str | None,
) -> str | None:
    if env_name in restore_env:
        value = restore_env[env_name]
        if value is None:
            return RESTORE_RUNTIME_ENV_DEFAULTS_WHEN_UNSET.get(env_name)
        return value
    return os.environ.get(env_name, fallback)


def _apply_restore_runtime_env_from_podinfo() -> dict[str, str | None]:
    podinfo_path = os.path.join(PODINFO_ROOT, RESTORE_RUNTIME_ENV_PODINFO_FILE)
    if not os.path.isfile(podinfo_path):
        return {}

    with open(podinfo_path, encoding="utf-8") as podinfo:
        payload = podinfo.read().strip()
    if not payload:
        return {}

    try:
        restore_env_payload = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid snapshot restore runtime env: {exc}") from exc

    env_config = restore_env_payload.get("env")
    if not isinstance(env_config, dict):
        raise RuntimeError("snapshot restore runtime env requires an object env field")

    applied = []
    cleared = []
    restored_env = {}
    for env_name, value in env_config.items():
        if env_name not in RESTORE_RUNTIME_ENV_NAMES:
            logger.warning("Ignoring unsupported snapshot restore env %s", env_name)
            continue
        if value is None:
            os.environ.pop(env_name, None)
            cleared.append(env_name)
            restored_env[env_name] = None
            continue
        if not isinstance(value, str):
            raise RuntimeError(
                f"snapshot restore runtime env {env_name} must be a string or null"
            )
        os.environ[env_name] = value
        applied.append(env_name)
        restored_env[env_name] = value

    logger.info(
        "Applied snapshot restore runtime env",
        extra={
            "applied_env": sorted(applied),
            "cleared_env": sorted(cleared),
        },
    )
    return restored_env
