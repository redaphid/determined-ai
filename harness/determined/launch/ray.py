import argparse
import logging
import os
import subprocess
import sys
import time
from typing import List, Tuple

import determined as det

RAY_PORT = 6379


def create_launch_cmd_head(
        proc_per_node: int, override_args: List[str]
) -> List[str]:
    cmd = [
        "ray",
        "start",
        "--head",
        "--port",
        str(RAY_PORT),
        "--num-gpus",
        str(proc_per_node),
    ]

    cmd.extend(override_args)
    return cmd


def create_launch_cmd_compute(
        proc_per_node: int, master_addr: str, override_args: List[str]
) -> List[str]:
    cmd = [
        "ray",
        "start",
        "--address",
        f"{master_addr}:{RAY_PORT}",
        "--num-gpus",
        str(proc_per_node),
    ]

    cmd.extend(override_args)
    return cmd


def create_log_redirect_cmd() -> List[str]:
    return [
        "python3",
        "-m",
        "determined.launch.wrap_rank",
        "RANK",
        "--",
    ]


def create_pid_server_cmd(allocation_id: str, num_slot_ids: int) -> List[str]:
    return [
        "python3",
        "-m",
        "determined.exec.pid_server",
        "--on-fail",
        "SIGTERM",
        "--on-exit",
        "WAIT",
        f"/tmp/pid_server-{allocation_id}",
        str(num_slot_ids),
        "--",
    ]


def create_pid_client_cmd(allocation_id: str) -> List[str]:
    return [
        "python3",
        "-m",
        "determined.exec.pid_client",
        f"/tmp/pid_server-{allocation_id}",
        "--",
    ]


def main(override_args: List[str], script: List[str]) -> int:
    override_args = override_args or []

    info = det.get_cluster_info()
    assert info is not None, "must be run on-cluster"

    single_slot = len(info.container_addrs) == 1 and len(info.slot_ids) == 1

    # Detect single-slot trials and skip distributed launch
    if single_slot:
        return subprocess.Popen(script).wait()

    os.environ["USE_RAY"] = "True"

    chief_ip = info.container_addrs[0]
    os.environ["DET_CHIEF_IP"] = chief_ip

    if info.container_rank > 0:
        ray_cmd = create_launch_cmd_compute(len(info.slot_ids), chief_ip, override_args)
    else:
        ray_cmd = create_launch_cmd_head(len(info.slot_ids), override_args)


    log_redirect_cmd = create_log_redirect_cmd()

    pid_server_cmd = create_pid_server_cmd(info.allocation_id, len(info.slot_ids))
    pid_client_cmd = create_pid_client_cmd(info.allocation_id)

    launch_cmd = pid_server_cmd + pid_client_cmd + log_redirect_cmd + script

    logging.info(f"Ray launching with: {launch_cmd}")
    print(f"ray cmd {ray_cmd}")
    if info.container_rank > 0:
        print(f"waiting for chief node")
        time.sleep(5)
        return subprocess.Popen(ray_cmd).wait()
    else:
        os.environ["RANK"] = "0"
        ray_proc = subprocess.Popen(ray_cmd)
        return subprocess.Popen(launch_cmd).wait()


def parse_args(args: List[str]) -> Tuple[List[str], List[str]]:
    if "--" in args:
        split = args.index("--")
        override_args = args[:split]
        args = args[split + 1 :]
    else:
        override_args = []

    parser = argparse.ArgumentParser(
        usage="%(prog)s [[TORCH_OVERRIDES...] --] (--trial TRIAL)|(SCRIPT...)",
        description=("Launch a script under pytorch distributed on a Determined cluster"),
        epilog=(
            "TORCH_OVERRIDES may be a list of arguments to pass directly to "
            "torch.distributed.launch to override the values set by Determined automatically.  "
            "When provided, the list of override arguments must be terminated by a `--` argument."
        ),
    )
    # For legacy Trial classes.
    parser.add_argument(
        "--trial",
        help=(
            "use a Trial class as the entrypoint to training.  When --trial is used, the SCRIPT "
            "positional argument must be omitted."
        ),
    )
    # For training scripts.
    parser.add_argument(
        "script",
        metavar="SCRIPT...",
        nargs=argparse.REMAINDER,
        help="script to launch for training",
    )
    parsed = parser.parse_args(args)

    script = parsed.script or []

    if parsed.trial is not None:
        if script:
            # When --trial is set, any other args are an error.
            parser.print_usage()
            print("error: extra arguments to --trial:", script, file=sys.stderr)
            sys.exit(1)
        script = det.util.legacy_trial_entrypoint_to_script(parsed.trial)
    elif not script:
        # There needs to be at least one script argument.
        parser.print_usage()
        print("error: empty script is not allowed", file=sys.stderr)
        sys.exit(1)

    return override_args, script


if __name__ == "__main__":
    override_args, script = parse_args(sys.argv[1:])
    print(f"override {override_args}, script {script}")
    sys.exit(main(override_args, script))
