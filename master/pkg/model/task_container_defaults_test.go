//nolint:exhaustivestruct
package model

import (
	"testing"

	"github.com/stretchr/testify/require"

	k8sV1 "k8s.io/api/core/v1"

	"github.com/determined-ai/determined/master/pkg/schemas/expconf"
)

func TestEnvironmentVarsDefaultMerging(t *testing.T) {
	gpuType := "tesla"
	pbsSlotsPerNode := 99
	defaults := &TaskContainerDefaultsConfig{
		EnvironmentVariables: &RuntimeItems{
			CPU:  []string{"cpu=default"},
			CUDA: []string{"cuda=default"},
			ROCM: []string{"rocm=default"},
		},
		Slurm: expconf.SlurmConfigV0{
			RawGpuType: &gpuType,
		},
		Pbs: expconf.PbsConfigV0{
			RawSlotsPerNode: &pbsSlotsPerNode,
		},
	}
	conf := expconf.ExperimentConfig{
		RawEnvironment: &expconf.EnvironmentConfig{
			RawEnvironmentVariables: &expconf.EnvironmentVariablesMap{
				RawCPU:  []string{"cpu=expconf"},
				RawCUDA: []string{"extra=expconf"},
			},
		},
	}
	defaults.MergeIntoExpConfig(&conf)

	require.Equal(t, conf.RawEnvironment.RawEnvironmentVariables,
		&expconf.EnvironmentVariablesMap{
			RawCPU:  []string{"cpu=default", "cpu=expconf"},
			RawCUDA: []string{"cuda=default", "extra=expconf"},
			RawROCM: []string{"rocm=default"},
		})

	require.Equal(t, *conf.RawSlurmConfig.RawGpuType, gpuType)
	require.Equal(t, *conf.RawPbsConfig.RawSlotsPerNode, pbsSlotsPerNode)
}

func TestPodSpecsDefaultMerging(t *testing.T) {
	defaults := &TaskContainerDefaultsConfig{
		CPUPodSpec: &k8sV1.Pod{
			Spec: k8sV1.PodSpec{
				SecurityContext: &k8sV1.PodSecurityContext{
					SELinuxOptions: &k8sV1.SELinuxOptions{
						Level: "cpuLevel",
						Role:  "cpuRole",
					},
				},
			},
		},
		GPUPodSpec: &k8sV1.Pod{
			Spec: k8sV1.PodSpec{
				SecurityContext: &k8sV1.PodSecurityContext{
					SELinuxOptions: &k8sV1.SELinuxOptions{
						Level: "gpuLevel",
						Role:  "gpuRole",
					},
				},
			},
		},
	}

	for i := 0; i <= 1; i++ {
		conf := expconf.ExperimentConfig{
			RawResources: &expconf.ResourcesConfig{RawSlotsPerTrial: &i},
			RawEnvironment: &expconf.EnvironmentConfig{
				RawPodSpec: &expconf.PodSpec{
					Spec: k8sV1.PodSpec{
						SecurityContext: &k8sV1.PodSecurityContext{
							SELinuxOptions: &k8sV1.SELinuxOptions{
								Level: "expconfLevel",
							},
						},
					},
				},
			},
		}
		defaults.MergeIntoExpConfig(&conf)

		expected := &expconf.PodSpec{
			Spec: k8sV1.PodSpec{
				SecurityContext: &k8sV1.PodSecurityContext{
					SELinuxOptions: &k8sV1.SELinuxOptions{
						Level: "expconfLevel",
						Role:  []string{"cpuRole", "gpuRole"}[i],
					},
				},
			},
		}
		require.Equal(t, expected, conf.RawEnvironment.RawPodSpec)
	}
}
