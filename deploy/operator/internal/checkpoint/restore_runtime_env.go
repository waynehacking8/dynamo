// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package checkpoint

import (
	"encoding/json"
	"fmt"

	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
)

// restoreRuntimeEnvNames is the small set of non-secret env vars that are read
// after CRIU restore, before or during DistributedRuntime/system-server setup.
// Do not add model-loading, CUDA/NCCL, credentials, or observability-only env
// here unless the restored process consumes them after restore and before it can
// serve.
var restoreRuntimeEnvNames = []string{
	// Parsed Python runtime config that must also refresh the in-memory config
	// passed to create_runtime().
	"DYN_DISCOVERY_BACKEND",
	"DYN_REQUEST_PLANE",
	"DYN_EVENT_PLANE",

	// DistributedRuntime infrastructure env read after restore.
	"NATS_SERVER",
	"ETCD_ENDPOINTS",

	// Runtime system server/readiness env read after restore.
	"DYN_SYSTEM_PORT",
	"DYN_HEALTH_CHECK_ENABLED",
	"DYN_SYSTEM_STARTING_HEALTH_STATUS",
	"DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS",
	"DYN_SYSTEM_HOST",
	"DYN_SYSTEM_HEALTH_PATH",
	"DYN_SYSTEM_LIVE_PATH",

	// Kubernetes discovery mode env read when the restored runtime registers.
	"DYN_KUBE_DISCOVERY_MODE",
	"CONTAINER_NAME",
}

type restoreRuntimeEnv struct {
	Env map[string]*string `json:"env"`
}

// ApplyRestoreRuntimeEnvAnnotation records non-secret restore-time runtime env
// on the Pod metadata so it can be projected through Downward API.
//
// CRIU restores the checkpoint-time process environment. The restored Dynamo
// process reads this annotation from /etc/podinfo before creating
// DistributedRuntime, allowing it to use the restore Pod's NATS/etcd/discovery
// and system-server configuration instead of stale checkpoint-job values.
func ApplyRestoreRuntimeEnvAnnotationForTargets(
	annotations map[string]string,
	podSpec *corev1.PodSpec,
	targets []string,
) error {
	if annotations == nil || podSpec == nil {
		return nil
	}
	if len(targets) == 0 {
		targets = []string{commonconsts.MainContainerName}
	}

	containers := make([]*corev1.Container, 0, len(targets))
	for _, name := range targets {
		var container *corev1.Container
		for i := range podSpec.Containers {
			if podSpec.Containers[i].Name == name {
				container = &podSpec.Containers[i]
				break
			}
		}
		if container == nil {
			return fmt.Errorf("checkpoint restore target %q does not exist in pod spec", name)
		}
		containers = append(containers, container)
	}

	config := restoreRuntimeEnv{Env: make(map[string]*string, len(restoreRuntimeEnvNames))}
	for _, name := range restoreRuntimeEnvNames {
		value, ok := commonLiteralEnvValue(containers, name)
		if ok {
			config.Env[name] = value
		}
	}

	payload, err := json.Marshal(config)
	if err != nil {
		return err
	}
	annotations[commonconsts.CheckpointRestoreRuntimeEnvAnnotation] = string(payload)
	return nil
}

func commonLiteralEnvValue(containers []*corev1.Container, name string) (*string, bool) {
	var common *string
	initialized := false
	for _, container := range containers {
		value, ok := literalEnvValue(container, name)
		if !ok {
			// Non-literal EnvVarSource values may refer to Secrets or Pod fields.
			// Do not copy them into annotations, and do not clear stale values
			// because the restore-time value is unknown to the operator.
			return nil, false
		}
		if !initialized {
			common = value
			initialized = true
			continue
		}
		if (value == nil) != (common == nil) {
			return nil, false
		}
		if value != nil && *value != *common {
			return nil, false
		}
	}
	return common, true
}

func literalEnvValue(container *corev1.Container, name string) (*string, bool) {
	for _, env := range container.Env {
		if env.Name != name {
			continue
		}
		if env.ValueFrom != nil {
			return nil, false
		}
		value := env.Value
		return &value, true
	}
	return nil, true
}
