package apptainer

import (
	"context"
	"syscall"

	"github.com/determined-ai/determined/agent/pkg/docker"
	"github.com/determined-ai/determined/agent/pkg/events"
	"github.com/determined-ai/determined/master/pkg/aproto"
	"github.com/determined-ai/determined/master/pkg/cproto"
	"github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/filters"
)

type ApptainerClient struct{}

func New() (ApptainerClient, error) {
	return ApptainerClient{}, nil
}

// CreateContainer implements container.ContainerRuntime
func (ApptainerClient) CreateContainer(ctx context.Context, req cproto.RunSpec, p events.Publisher[docker.Event]) (string, error) {
	panic("unimplemented")
}

// ListRunningContainers implements container.ContainerRuntime
func (ApptainerClient) ListRunningContainers(ctx context.Context, fs filters.Args) (map[cproto.ID]types.Container, error) {
	panic("unimplemented")
}

// PullImage implements container.ContainerRuntime
func (ApptainerClient) PullImage(ctx context.Context, req docker.PullImage, p events.Publisher[docker.Event]) error {
	panic("unimplemented")
}

// ReattachContainer implements container.ContainerRuntime
func (ApptainerClient) ReattachContainer(ctx context.Context, filter filters.Args) (*docker.Container, *aproto.ExitCode, error) {
	panic("unimplemented")
}

// RemoveContainer implements container.ContainerRuntime
func (ApptainerClient) RemoveContainer(ctx context.Context, id string, force bool) error {
	panic("unimplemented")
}

// RunContainer implements container.ContainerRuntime
func (ApptainerClient) RunContainer(ctx context.Context, waitCtx context.Context, id string) (*docker.Container, error) {
	panic("unimplemented")
}

// SignalContainer implements container.ContainerRuntime
func (ApptainerClient) SignalContainer(ctx context.Context, id string, sig syscall.Signal) error {
	panic("unimplemented")
}
