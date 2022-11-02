import sys
from argparse import ONE_OR_MORE, FileType, Namespace
from functools import partial
from pathlib import Path
from typing import Any, List

from termcolor import colored

from determined import cli
from determined.cli import command, task
from determined.common import api, context
from determined.common.api import authentication, request
from determined.common.check import check_eq
from determined.common.declarative_argparse import Arg, Cmd, Group


@authentication.required
def start_tensorboard(args: Namespace) -> None:
    if not (args.trial_ids or args.experiment_ids):
        print("Either experiment_ids or trial_ids must be specified.")
        sys.exit(1)

    config = command.parse_config(args.config_file, None, args.config, [])
    req_body = {
        "config": config,
        "trial_ids": args.trial_ids,
        "experiment_ids": args.experiment_ids,
    }

    req_body["files"] = context.read_legacy_context(args.context, args.include)

    api_resp = api.post(args.master, "api/v1/tensorboards", json=req_body).json()
    maxSlotsExceeded = api_resp["maxCurrentSlotsExceeded"]
    resp = api_resp["tensorboard"]

    if args.detach:
        print(resp["id"])
        return

    url = "tensorboard/{}/events".format(resp["id"])
    with api.ws(args.master, url) as ws:
        for msg in ws:
            if msg["log_event"] is not None:
                # TensorBoard will print a url by default. The URL is incorrect since
                # TensorBoard is not aware of the master proxy address it is assigned.
                if "http" in msg["log_event"]:
                    continue

            if msg["service_ready_event"]:
                if args.no_browser:
                    url = api.make_url(args.master, resp["serviceAddress"])
                else:
                    url = api.browser_open(
                        args.master,
                        request.make_interactive_task_url(
                            task_id=resp["id"],
                            service_address=resp["serviceAddress"],
                            resource_pool=resp["resourcePool"],
                            description=resp["description"],
                            task_type="tensorboard",
                            maxSlotsExceeded= api_resp["maxCurrentSlotsExceeded"]
                        ),
                    )
                if maxSlotsExceeded:
                    warning = ("The requested job requires more slots than currently available. ", 
                    "You may need to increase cluster resources in order for the job to run." ) 
                    print(colored(warning), "yellow")
                print(colored("TensorBoard is running at: {}".format(url), "green"))
                command.render_event_stream(msg)
                break
            command.render_event_stream(msg)


@authentication.required
def open_tensorboard(args: Namespace) -> None:
    tensorboard_id = command.expand_uuid_prefixes(args)
    resp = api.get(args.master, "api/v1/tensorboards/{}".format(tensorboard_id)).json()[
        "tensorboard"
    ]
    check_eq(resp["state"], "STATE_RUNNING", "TensorBoard must be in a running state")
    api.browser_open(
        args.master,
        request.make_interactive_task_url(
            task_id=resp["id"],
            service_address=resp["serviceAddress"],
            resource_pool=resp["resourcePool"],
            description=resp["description"],
            task_type="tensorboard",
            maxSlotsExceeded=False
        ),
    )


# fmt: off

args_description = [
    Cmd("tensorboard", None, "manage TensorBoard instances", [
        Cmd("list ls", partial(command.list_tasks), "list TensorBoard instances", [
            Arg("-q", "--quiet", action="store_true",
                help="only display the IDs"),
            Arg("--all", "-a", action="store_true",
                help="show all TensorBoards (including other users')"),
            Group(cli.output_format_args["json"], cli.output_format_args["csv"]),
        ], is_default=True),
        Cmd("start", start_tensorboard, "start new TensorBoard instance", [
            Arg("experiment_ids", type=int, nargs="*",
                help="experiment IDs to load into TensorBoard. At most 100 trials from "
                     "the specified experiment will be loaded into TensorBoard. If the "
                     "experiment has more trials, the 100 best-performing trials will "
                     "be used."),
            Arg("-t", "--trial-ids", nargs=ONE_OR_MORE, type=int,
                help="trial IDs to load into TensorBoard; at most 100 trials are "
                     "allowed per TensorBoard instance"),
            Arg("--config-file", default=None, type=FileType("r"),
                help="command config file (.yaml)"),
            Arg("-c", "--context", default=None, type=Path, help=command.CONTEXT_DESC),
            Arg(
                "-i",
                "--include",
                default=[],
                action="append",
                type=Path,
                help=command.INCLUDE_DESC
            ),
            Arg("--config", action="append", default=[], help=command.CONFIG_DESC),
            Arg("--no-browser", action="store_true",
                help="don't open TensorBoard in a browser after startup"),
            Arg("-d", "--detach", action="store_true",
                help="run in the background and print the ID")
        ]),
        Cmd("config", partial(command.config),
            "display TensorBoard config", [
                Arg("tensorboard_id", type=str, help="TensorBoard ID")
        ]),
        Cmd("open", open_tensorboard,
            "open existing TensorBoard instance", [
                Arg("tensorboard_id", help="TensorBoard ID")
            ]),
        Cmd("logs", partial(task.logs),
            "fetch TensorBoard instance logs", [
            Arg("task_id", help="TensorBoard ID", metavar="tensorboard_id"),
            *task.common_log_options,
        ]),
        Cmd("kill", partial(command.kill), "kill TensorBoard instance", [
            Arg("tensorboard_id", help="TensorBoard ID", nargs=ONE_OR_MORE),
            Arg("-f", "--force", action="store_true", help="ignore errors"),
        ]),
        Cmd("set", None, "set TensorBoard attributes", [
            Cmd("priority", partial(command.set_priority), "set TensorBoard priority", [
                Arg("tensorboard_id", help="TensorBoard ID"),
                Arg("priority", type=int, help="priority"),
            ]),
        ]),
    ])
]  # type: List[Any]

# fmt: on
