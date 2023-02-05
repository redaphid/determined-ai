package singularity

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/davecgh/go-spew/spew"
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
	stateCache     = "/var/cache/determined/singularity_containers.json"
	stateCacheCopy = "/var/cache/determined/singularity_containers.json.copy"
	cleanupDelay   = time.Hour
)

type SingularityClient struct {
	log        *logrus.Entry
	mu         sync.Mutex
	wg         waitgroupx.Group
	containers map[cproto.ID]*SingularityContainer // TODO: Snapshot this
}

type SingularityContainer struct {
	PID         int                    `json:"pid"`
	Cmd         []string               `json:"cmd"`
	Req         cproto.RunSpec         `json:"req"`
	NetworkMode dcontainer.NetworkMode `json:"network_mode"`
	Ports       nat.PortSet            `json:"ports"`

	Proc *os.Process `json:"-"`
}

func New() (*SingularityClient, error) {
	cl := &SingularityClient{
		log:        logrus.WithField("compotent", "singularity"),
		wg:         waitgroupx.WithContext(context.Background()),
		containers: make(map[cproto.ID]*SingularityContainer),
	}

	if err := cl.LoadCache(); err != nil {
		return nil, fmt.Errorf("initial cache load: %w", err)
	}
	return cl, nil
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
	args = append(args, "--pwd", req.ContainerConfig.WorkingDir)
	args = append(args, "--env", "DET_NO_FLUENT=true")
	for _, env := range req.ContainerConfig.Env {
		args = append(args, "--env", env)
	}

	tmpdir, err := os.MkdirTemp("/var/tmp", fmt.Sprintf("*-%s", id)) // TODO: cleanup
	if err != nil {
		return "", fmt.Errorf("making tmp dir for archives: %w", err)
	}
	for _, a := range req.Archives {
		if err := archive.Write(filepath.Join(tmpdir, a.Path), a.Archive); err != nil {
			return "", fmt.Errorf("writing archive for %s: %w", a.Path, err)
		}
	}
	// HACK: can't just mount top level stuff because then you override /opt and there is no
	// functioning python installation, algorithm that works is like "mount the top lvl dirs except
	// when it would fuck something up, then try to mount lower".
	for _, dst := range []string{"/run/determined", "/opt/determined", "/etc/ssh"} {
		src := filepath.Join(tmpdir, dst)
		if _, err := os.Stat(src); err == nil {
			args = append(args, "--bind", fmt.Sprintf("%s:%s", src, dst))
		}
	}

	// TODO: device mappings and stuff for amd.
	for _, d := range req.HostConfig.DeviceRequests {
		if d.Driver == "nvidia" {
			args = append(args, "--nv")
			break
		}
	}

	args = append(args, req.ContainerConfig.Image)
	args = append(args, req.ContainerConfig.Cmd...)
	s.log.Trace(fmt.Sprintf("singularity %s", strings.Join(args, " ")))

	s.mu.Lock()
	defer s.mu.Unlock()
	s.containers[id] = &SingularityContainer{
		Cmd:         append([]string{"singularity"}, args...),
		Req:         req,
		NetworkMode: "host",
		Ports:       req.ContainerConfig.ExposedPorts,
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

	cmd := exec.CommandContext(waitCtx, cont.Cmd[0], cont.Cmd[1:]...)
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
	cont.Proc = cmd.Process

	wchan := make(chan dcontainer.ContainerWaitOKBody)
	errchan := make(chan error)
	s.wg.Go(func(ctx context.Context) {
		var body dcontainer.ContainerWaitOKBody
		if err := cmd.Wait(); err != nil {
			body.Error = &dcontainer.ContainerWaitOKBodyError{Message: err.Error()}
		}

		select {
		case wchan <- body:
		case <-ctx.Done():
		}

		s.mu.Lock()
		defer s.mu.Unlock()
		delete(s.containers, cproto.ID(id))
	})

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
		ContainerWaiter: docker.ContainerWaiter{Waiter: wchan, Errs: errchan},
	}, nil
}

// ReattachContainer implements container.ContainerRuntime
func (s *SingularityClient) ReattachContainer(
	ctx context.Context,
	reattachID cproto.ID,
) (*docker.Container, *aproto.ExitCode, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	var cont *SingularityContainer
	for id, rcont := range s.containers {
		if reattachID != id {
			continue
		}
		cont = rcont
		break
	}
	if cont == nil {
		return nil, nil, container.ErrMissing
	}

	wchan := make(chan dcontainer.ContainerWaitOKBody)
	errchan := make(chan error)
	s.wg.Go(func(ctx context.Context) {
		state, err := cont.Proc.Wait()
		spew.Dump(*state, state.ExitCode(), err)
		if err != nil {
			select {
			case errchan <- err:
			case <-ctx.Done():
				return
			}
		}

		var body dcontainer.ContainerWaitOKBody
		if code := state.ExitCode(); code != 0 {
			body.StatusCode = int64(code)
			body.Error = &dcontainer.ContainerWaitOKBodyError{Message: state.String()}
		}

		select {
		case wchan <- body:
		case <-ctx.Done():
		}

		s.mu.Lock()
		defer s.mu.Unlock()
		delete(s.containers, reattachID)
	})

	return &docker.Container{
		ContainerInfo: types.ContainerJSON{
			ContainerJSONBase: &types.ContainerJSONBase{
				ID: strconv.Itoa(cont.Proc.Pid),
				HostConfig: &dcontainer.HostConfig{
					NetworkMode: cont.NetworkMode,
				},
			},
			Config: &dcontainer.Config{
				ExposedPorts: nat.PortSet{},
			},
		}, // TODO
		ContainerWaiter: docker.ContainerWaiter{Waiter: wchan, Errs: errchan},
	}, nil, nil
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
	for id := range s.containers {
		resp[id] = types.Container{} // TODO
	}
	return resp, nil
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

		line = strings.TrimPrefix(line, "FATAL:   ") // TODO: prase out levels everywhere, sometimes convert.
		p.Publish(ctx, docker.NewLogEvent(model.LogLevelInfo, line))
	}
	return nil
}

func (s *SingularityClient) LoadCache() error {
	f, err := os.Open(stateCache)
	switch {
	case errors.Is(err, os.ErrNotExist):
		return nil
	case err != nil:
		return fmt.Errorf("opening state cache: %w", err)
	}

	if err := json.NewDecoder(f).Decode(&s.containers); err != nil {
		return fmt.Errorf("decoding state cache: %w", err)
	}
	return nil
}

func (s *SingularityClient) PersistCache() error {
	bs, err := json.Marshal(s.containers)
	if err != nil {
		return fmt.Errorf("persisting cache: %w", err)
	}

	f, err := os.OpenFile(stateCacheCopy, os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return fmt.Errorf("opening state cache copy: %w", err)
	}

	n, err := f.Write(bs)
	switch {
	case err != nil:
		return fmt.Errorf("writing state cache: %w", err)
	case n != len(bs):
		return fmt.Errorf("unable to write full cache (%d != %d)", n, len(bs))
	}

	if err := os.Rename(stateCacheCopy, stateCache); err != nil {
		return fmt.Errorf("commiting state cache: %w", err)
	}
	return nil
}
