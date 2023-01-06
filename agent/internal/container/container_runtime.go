package container

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

type ContainerRuntime interface {
	ReattachContainer(
		ctx context.Context,
		id cproto.ID,
	) (*docker.Container, *aproto.ExitCode, error)

	PullImage(ctx context.Context, req docker.PullImage, p events.Publisher[docker.Event]) error

	CreateContainer(
		ctx context.Context,
		id cproto.ID,
		req cproto.RunSpec,
		p events.Publisher[docker.Event],
	) (string, error)

	RunContainer(
		ctx context.Context,
		waitCtx context.Context,
		id string,
		p events.Publisher[docker.Event],
	) (*docker.Container, error)

	SignalContainer(ctx context.Context, id string, sig syscall.Signal) error

	RemoveContainer(ctx context.Context, id string, force bool) error

	ListRunningContainers(ctx context.Context, fs filters.Args) (map[cproto.ID]types.Container, error)
}
