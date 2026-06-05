# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for snapshot restore config helpers."""

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SNAPSHOT_PY = _REPO_ROOT / "components/src/dynamo/common/utils/snapshot.py"


def _load_snapshot_module():
    """Load snapshot.py without importing the native dynamo package."""
    stub_names = (
        "dynamo",
        "dynamo.common",
        "dynamo.common.utils",
        "dynamo.common.utils.namespace",
    )
    previous_modules = {name: sys.modules.get(name) for name in stub_names}

    dynamo_stub = types.ModuleType("dynamo")
    common_stub = types.ModuleType("dynamo.common")
    utils_stub = types.ModuleType("dynamo.common.utils")
    namespace_stub = types.ModuleType("dynamo.common.utils.namespace")

    def get_worker_namespace(namespace=None):
        import os

        namespace = namespace or os.environ.get("DYN_NAMESPACE", "dynamo")
        suffix = os.environ.get("DYN_NAMESPACE_WORKER_SUFFIX")
        if suffix:
            return f"{namespace}-{suffix}"
        return namespace

    namespace_stub.get_worker_namespace = get_worker_namespace

    sys.modules["dynamo"] = dynamo_stub
    sys.modules["dynamo.common"] = common_stub
    sys.modules["dynamo.common.utils"] = utils_stub
    sys.modules["dynamo.common.utils.namespace"] = namespace_stub

    spec = importlib.util.spec_from_file_location(
        "_dynamo_snapshot_under_test",
        _SNAPSHOT_PY,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        for name, previous in previous_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return module


snapshot = _load_snapshot_module()


@pytest.fixture(autouse=True)
def clean_restore_env(monkeypatch):
    env_names = set(snapshot.RESTORE_RUNTIME_ENV_NAMES)
    env_names.update(snapshot.KUBERNETES_REQUIRED_PODINFO_FILES)
    env_names.update(snapshot.KUBERNETES_OPTIONAL_PODINFO_FILES)
    for name in env_names:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture()
def podinfo_root(tmp_path, monkeypatch):
    monkeypatch.setattr(snapshot, "PODINFO_ROOT", str(tmp_path))
    return tmp_path


def _write_podinfo(podinfo_root: Path, name: str, value: str) -> None:
    (podinfo_root / name).write_text(value, encoding="utf-8")


def test_restore_runtime_env_names_are_minimal():
    assert {
        "DYN_DISCOVERY_BACKEND",
        "DYN_REQUEST_PLANE",
        "DYN_EVENT_PLANE",
        "NATS_SERVER",
        "ETCD_ENDPOINTS",
        "DYN_SYSTEM_PORT",
        "DYN_HEALTH_CHECK_ENABLED",
        "DYN_SYSTEM_STARTING_HEALTH_STATUS",
        "DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS",
        "DYN_SYSTEM_HOST",
        "DYN_SYSTEM_HEALTH_PATH",
        "DYN_SYSTEM_LIVE_PATH",
        "DYN_KUBE_DISCOVERY_MODE",
        "CONTAINER_NAME",
    }.issubset(snapshot.RESTORE_RUNTIME_ENV_NAMES)

    assert "MODEL_EXPRESS_URL" not in snapshot.RESTORE_RUNTIME_ENV_NAMES
    assert "PROMETHEUS_ENDPOINT" not in snapshot.RESTORE_RUNTIME_ENV_NAMES
    assert "DYN_SYSTEM_ENABLED" not in snapshot.RESTORE_RUNTIME_ENV_NAMES
    assert snapshot.RESTORE_RUNTIME_ENV_DEFAULTS_WHEN_UNSET == {
        "DYN_DISCOVERY_BACKEND": "etcd",
        "DYN_REQUEST_PLANE": "tcp",
        "DYN_EVENT_PLANE": None,
    }


def test_reload_restore_config_refreshes_runtime_env(podinfo_root, monkeypatch):
    _write_podinfo(podinfo_root, "dyn_namespace", "restore-ns")
    _write_podinfo(podinfo_root, "dyn_namespace_worker_suffix", "abc123")
    _write_podinfo(podinfo_root, "dyn_component", "decode")
    _write_podinfo(podinfo_root, "dyn_parent_dgd_k8s_name", "llama")
    _write_podinfo(podinfo_root, "dyn_parent_dgd_k8s_namespace", "serving")
    _write_podinfo(
        podinfo_root,
        snapshot.RESTORE_RUNTIME_ENV_PODINFO_FILE,
        json.dumps(
            {
                "env": {
                    "DYN_DISCOVERY_BACKEND": "kubernetes",
                    "DYN_REQUEST_PLANE": "nats",
                    "DYN_EVENT_PLANE": "zmq",
                    "DYN_SYSTEM_PORT": "9090",
                    "NATS_SERVER": "nats://nats:4222",
                    "ETCD_ENDPOINTS": None,
                }
            }
        ),
    )
    monkeypatch.setenv("ETCD_ENDPOINTS", "http://checkpoint-etcd:2379")

    restored = snapshot.reload_snapshot_restore_config(
        namespace="checkpoint-ns",
        discovery_backend="etcd",
        request_plane="tcp",
        event_plane=None,
    )

    assert restored.namespace == "restore-ns-abc123"
    assert restored.discovery_backend == "kubernetes"
    assert restored.request_plane == "nats"
    assert restored.event_plane == "zmq"
    assert snapshot.os.environ["DYN_SYSTEM_PORT"] == "9090"
    assert snapshot.os.environ["NATS_SERVER"] == "nats://nats:4222"
    assert "ETCD_ENDPOINTS" not in snapshot.os.environ


def test_reload_restore_config_does_not_require_kubernetes_podinfo_for_etcd(
    podinfo_root,
):
    _write_podinfo(
        podinfo_root,
        snapshot.RESTORE_RUNTIME_ENV_PODINFO_FILE,
        json.dumps(
            {
                "env": {
                    "DYN_DISCOVERY_BACKEND": "etcd",
                    "DYN_REQUEST_PLANE": None,
                    "DYN_EVENT_PLANE": None,
                    "ETCD_ENDPOINTS": "http://restore-etcd:2379",
                    "NATS_SERVER": None,
                }
            }
        ),
    )

    restored = snapshot.reload_snapshot_restore_config(
        namespace="checkpoint-ns",
        discovery_backend="kubernetes",
        request_plane="nats",
        event_plane="nats",
    )

    assert restored.namespace == "checkpoint-ns"
    assert restored.discovery_backend == "etcd"
    assert restored.request_plane == "tcp"
    assert restored.event_plane is None
    assert snapshot.os.environ["ETCD_ENDPOINTS"] == "http://restore-etcd:2379"
    assert "NATS_SERVER" not in snapshot.os.environ
