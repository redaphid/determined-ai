package singularity

import (
	"context"
	"testing"

	"github.com/determined-ai/determined/agent/pkg/docker"
	"github.com/determined-ai/determined/agent/pkg/events"
	"github.com/determined-ai/determined/master/pkg/cproto"
	"github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/api/types/network"
	"github.com/docker/docker/api/types/strslice"
	"github.com/sirupsen/logrus"
	"github.com/stretchr/testify/require"
)

func TestSingularity(t *testing.T) {
	logrus.SetLevel(logrus.TraceLevel)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	t.Log("creating client")
	cl, err := New()
	require.NoError(t, err)

	t.Log("pulling container image")
	image := "docker://determinedai/environments:py-3.8-pytorch-1.10-tf-2.8-cpu-24586f0"
	cprotoID := cproto.NewID()
	evs := make(chan docker.Event, 1024)
	pub := events.ChannelPublisher(evs)
	err = cl.PullImage(ctx, docker.PullImage{
		Name:     image,
		Registry: &types.AuthConfig{},
	}, pub)
	require.NoError(t, err)

	t.Log("creating container")
	id, err := cl.CreateContainer(
		ctx,
		cprotoID,
		cproto.RunSpec{
			ContainerConfig: container.Config{
				Image: image,
				Cmd:   strslice.StrSlice{"/run/determined/train/entrypoint.sh"},
				Env:   []string{"DET_NO_FLUENT=true"},
			},
			HostConfig:       container.HostConfig{},
			NetworkingConfig: network.NetworkingConfig{},
			Archives:         []cproto.RunArchive{},
			UseFluentLogging: false,
		},
		pub,
	)
	require.NoError(t, err)

	t.Log("running container")
	waiter, err := cl.RunContainer(ctx, ctx, id, pub)
	require.NoError(t, err)

	select {
	case res := <-waiter.ContainerWaiter.Waiter:
		require.Nil(t, res.Error)
	case <-ctx.Done():
	}
}
