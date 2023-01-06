package singularity

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/determined-ai/determined/agent/internal/container"
	"github.com/determined-ai/determined/agent/pkg/docker"
	"github.com/determined-ai/determined/agent/pkg/events"
	"github.com/determined-ai/determined/master/pkg/aproto"
	"github.com/determined-ai/determined/master/pkg/archive"
	"github.com/determined-ai/determined/master/pkg/cproto"
	"github.com/determined-ai/determined/master/pkg/model"
	"github.com/determined-ai/determined/master/pkg/syncx/waitgroupx"
	"github.com/docker/docker/api/types"
	dcontainer "github.com/docker/docker/api/types/container"
	"github.com/docker/docker/api/types/filters"
	"github.com/docker/go-connections/nat"
	"github.com/sirupsen/logrus"
)

const (
	containerArchivePath         = "/determined-archives"
	runDir                       = "/run/determined"
	singularityWrapperEntrypoint = "singularity-entrypoint-wrapper.sh"
	cleanupDelay                 = time.Hour
)

type SingularityClient struct {
	log        *logrus.Entry
	mu         sync.Mutex
	wg         waitgroupx.Group
	containers map[cproto.ID]*SingularityContainer
}

type SingularityContainer struct {
	PID         int                    `json:"pid"`
	Cmd         []string               `json:"cmd"`
	Req         cproto.RunSpec         `json:"req"`
	NetworkMode dcontainer.NetworkMode `json:"network_mode"`
	Ports       nat.PortSet            `json:"ports"`
	TmpDir      string                 `json:"tmp_dir"`

	Proc *os.Process `json:"-"`
}

func New() (*SingularityClient, error) {
	return &SingularityClient{
		log:        logrus.WithField("compotent", "singularity"),
		wg:         waitgroupx.WithContext(context.Background()),
		containers: make(map[cproto.ID]*SingularityContainer),
	}, nil
}

// PullImage implements container.ContainerRuntime
func (s *SingularityClient) PullImage(ctx context.Context, req docker.PullImage, p events.Publisher[docker.Event]) error {
	if err := p.Publish(ctx, docker.NewBeginStatsEvent(docker.ImagePullStatsKind)); err != nil {
		return err
	}
	defer func() {
		if scErr := p.Publish(ctx, docker.NewEndStatsEvent(docker.ImagePullStatsKind)); scErr != nil {
			s.log.WithError(scErr).Warn("did not send image pull done stats")
		}
	}()

	args := []string{"pull"}
	if req.ForcePull {
		args = append(args, "--force")
	}
	args = append(args, req.Name)
	s.log.Tracef("singularity %s", strings.Join(args, " "))

	cmd := exec.CommandContext(ctx, "singularity", args...)

	output, err := cmd.CombinedOutput() // TODO: stream pull logs
	switch {
	case strings.Contains(string(output), "Image file already exists"):
		break
	case err != nil:
		return fmt.Errorf("pulling singularity image: %w\n%s", err, string(output))
	}

	for _, line := range strings.Split(string(output), "\n") {
		if len(strings.TrimSpace(line)) == 0 {
			continue
		}

		line = strings.TrimPrefix(line, "FATAL:   ") // TODO: parse out levels everywhere, sometimes convert.
		p.Publish(ctx, docker.NewLogEvent(model.LogLevelInfo, line))
	}
	return nil
}

// CreateContainer implements container.ContainerRuntime
func (s *SingularityClient) CreateContainer(
	ctx context.Context,
	id cproto.ID,
	req cproto.RunSpec,
	p events.Publisher[docker.Event],
) (string, error) {
	var args []string
	args = append(args, "run")
	args = append(args, "--writable-tmpfs")
	args = append(args, "--env", fmt.Sprintf("DET_WORKDIR=%s", req.ContainerConfig.WorkingDir))
	args = append(args, "--env", "DET_NO_FLUENT=true")
	args = append(args, "--env", "DET_UNPACK_ARCHIVES=true")
	for _, env := range req.ContainerConfig.Env {
		args = append(args, "--env", env)
	}

	tmpdir, err := os.MkdirTemp("/var/tmp", fmt.Sprintf("*-%s", id)) // TODO: cleanup
	if err != nil {
		return "", fmt.Errorf("making tmp dir for archives: %w", err)
	}

	req.Archives = append(req.Archives)

	for _, a := range req.Archives {
		src := filepath.Join(tmpdir, a.Path)
		if err := archive.Write(src, a.Archive); err != nil {
			return "", fmt.Errorf("writing archive for %s: %w", a.Path, err)
		}
	}
	args = append(args, "--bind", fmt.Sprintf("%s:%s", tmpdir, containerArchivePath))

	// TODO: device mappings and stuff for amd.
	for _, d := range req.HostConfig.DeviceRequests {
		if d.Driver == "nvidia" {
			args = append(args, "--nv")
			break
		}
	}

	args = append(args, req.ContainerConfig.Image)
	args = append(args, path.Join(containerArchivePath, runDir, singularityWrapperEntrypoint))
	args = append(args, req.ContainerConfig.Cmd...)

	s.log.Info("singularity \\")
	toPrint := ""
	for _, arg := range args {
		if strings.HasPrefix(arg, "--") {
			toPrint += " \\\n"
			toPrint += "\t"
			toPrint += arg
		} else {
			toPrint += " "
			toPrint += arg
		}
	}
	s.log.Info(toPrint)

	s.mu.Lock()
	defer s.mu.Unlock()
	s.containers[id] = &SingularityContainer{
		Cmd:         append([]string{"singularity"}, args...),
		Req:         req,
		NetworkMode: "host",
		Ports:       req.ContainerConfig.ExposedPorts,
		TmpDir:      tmpdir,
	}
	return id.String(), nil
}

// RunContainer implements container.ContainerRuntime
func (s *SingularityClient) RunContainer(
	ctx context.Context,
	waitCtx context.Context,
	id string,
	p events.Publisher[docker.Event],
) (*docker.Container, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	var cont *SingularityContainer
	for cID, rcont := range s.containers {
		if cproto.ID(id) != cID {
			continue
		}
		cont = rcont
		break
	}
	if cont == nil {
		return nil, container.ErrMissing
	}

	cmd := exec.CommandContext(waitCtx, cont.Cmd[0], cont.Cmd[1:]...) // TODO: no command ctx, recover stuff
	stdout, oerr := cmd.StdoutPipe()
	stderr, eerr := cmd.StderrPipe()
	if oerr != nil || eerr != nil {
		s.log.Error(oerr.Error(), eerr.Error())
	} else {
		s.wg.Go(func(ctx context.Context) {
			for scan := bufio.NewScanner(stdout); scan.Scan(); {
				p.Publish(ctx, docker.NewLogEvent(model.LogLevelInfo, scan.Text())) // TODO: stdtype
			}
		})
		s.wg.Go(func(ctx context.Context) {
			for scan := bufio.NewScanner(stderr); scan.Scan(); {
				p.Publish(ctx, docker.NewLogEvent(model.LogLevelInfo, scan.Text())) // TODO: stdtype
			}
		})
	}

	// TODO: device mappings and stuff for amd.
	var devices string
	for _, d := range cont.Req.HostConfig.DeviceRequests {
		if d.Driver == "nvidia" {
			devices = strings.Join(d.DeviceIDs, ",")
		}
	}
	cmd.Env = append(cmd.Env,
		fmt.Sprintf("SINGULARITYENV_CUDA_VISIBLE_DEVICES=%s", devices),
		fmt.Sprintf("APPTAINERENV_CUDA_VISIBLE_DEVICES=%s", devices),
	)

	cmd.Env = append(cmd.Env, fmt.Sprintf("PATH=%s", os.Getenv("PATH"))) // TODO: without this, --nv doesn't work right.

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("starting singularity container: %w", err)
	}
	cont.PID = cmd.Process.Pid
	cont.Proc = cmd.Process
	s.log.Infof("started container %s with pid %d", id, cont.PID)

	return &docker.Container{
		ContainerInfo: types.ContainerJSON{
			ContainerJSONBase: &types.ContainerJSONBase{
				ID: strconv.Itoa(cont.Proc.Pid),
				HostConfig: &dcontainer.HostConfig{
					NetworkMode: cont.NetworkMode,
				},
			},
			Config: &dcontainer.Config{
				ExposedPorts: cont.Ports,
			},
		}, // TODO
		ContainerWaiter: s.WaitOnContainer(cproto.ID(id), cont),
	}, nil
}

// ReattachContainer implements container.ContainerRuntime
func (s *SingularityClient) ReattachContainer(
	ctx context.Context,
	reattachID cproto.ID,
) (*docker.Container, *aproto.ExitCode, error) {
	return nil, nil, container.ErrMissing
}

// RemoveContainer implements container.ContainerRuntime
func (s *SingularityClient) RemoveContainer(ctx context.Context, id string, force bool) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	cont, ok := s.containers[cproto.ID(id)]
	if !ok {
		return container.ErrMissing
	}
	return cont.Proc.Kill()
}

// SignalContainer implements container.ContainerRuntime
func (s *SingularityClient) SignalContainer(ctx context.Context, id string, sig syscall.Signal) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	cont, ok := s.containers[cproto.ID(id)]
	if !ok {
		return container.ErrMissing
	}
	return cont.Proc.Signal(sig)
}

// ListRunningContainers implements container.ContainerRuntime
func (s *SingularityClient) ListRunningContainers(ctx context.Context, fs filters.Args) (map[cproto.ID]types.Container, error) {
	resp := make(map[cproto.ID]types.Container)

	s.mu.Lock()
	defer s.mu.Unlock()
	for id, cont := range s.containers {
		resp[id] = types.Container{
			ID:     string(id),
			Labels: cont.Req.ContainerConfig.Labels,
		}
	}
	return resp, nil
}

func (s *SingularityClient) WaitOnContainer(
	id cproto.ID,
	cont *SingularityContainer,
) docker.ContainerWaiter {
	wchan := make(chan dcontainer.ContainerWaitOKBody, 1)
	errchan := make(chan error)
	s.wg.Go(func(ctx context.Context) {
		defer close(wchan)
		defer close(errchan)

		var body dcontainer.ContainerWaitOKBody
		switch state, err := cont.Proc.Wait(); {
		case ctx.Err() != nil && err == nil && state.ExitCode() == -1:
			s.log.Trace("detached from container process")
			return
		case err != nil:
			s.log.Tracef("proc %d for container %s exited: %s", cont.PID, id, err)
			s.log.Tracef("proc state: %s", cont.Proc)
			body.Error = &dcontainer.ContainerWaitOKBodyError{Message: err.Error()}
		default:
			s.log.Tracef("proc %s for container %d exited with %d", cont.PID, id, state.ExitCode())
			body.StatusCode = int64(state.ExitCode())
		}

		select {
		case wchan <- body:
		case <-ctx.Done():
			return
		}

		s.mu.Lock()
		defer s.mu.Unlock()
		s.log.Tracef("forgetting completed container: %s", id)
		delete(s.containers, cproto.ID(id))
		if err := os.RemoveAll(cont.TmpDir); err != nil {
			s.log.WithError(err).Error("failed to cleanup tmpdir (ephemeral mounts, etc)")
		}
	})
	return docker.ContainerWaiter{Waiter: wchan, Errs: errchan}
}
