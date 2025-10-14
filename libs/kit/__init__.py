""" Includes the main functions to initialize the kit framework for deep learning projects. """

import os
import pdb
import shlex
import argparse
import logging

from tqdm.auto import tqdm
import yaml

import kit.globals as G
from kit.path import join
from kit.data import file_to_str, str_to_file, DD
from kit.jobs import get_next_job_id
from kit.log import setup_logger, log_info, log_warning
from kit.time import now


tqdm.pandas()

DEVICE = "cpu"
MODEL = None

JOB_INFO = DD()


def set_args(arg_string=None, docstring=None, argparser=None):
    """sets the arguments for the program

    The arguments are stored in G.ARGS and G.ARGS_UNKNOWN

    :param arg_string: string with arguments (project, env, task, job, debug)
        this is optional, if not specified, the command line arguments are used
    :param docstring: docstring of the main program. Will be used for the help message
    :param argparser: if not None, this argparser is used instead of the default one
        this allows to add additional arguments to the program
    """

    # add standard arguments
    if argparser is None:
        argparser = argparse.ArgumentParser(
            description=docstring, formatter_class=argparse.RawTextHelpFormatter
        )
    argparser.add_argument(
        "--task",
        type=str,
        default="",
        help="""task configuration file without extension. It is in YAML format and can be
found in configs/CAPE-XVAE/tasks/
Most importantly, it specifies the handler to be used for the task.
The handler is a python file that can be found in libs/CAPE/XVAE/handler/""",
    )
    argparser.add_argument("--job", type=str, default="AUTO")
    argparser.add_argument(
        "--domain",
        type=str,
        default="d",
        help="allows to keep designs separate (e.g. 'GFP')",
    )
    argparser.add_argument(
        "--env",
        type=str,
        default="py",
        help="""environment configuration file without extension. It is in YAML format and
can be found in env/. These yaml files specify the directory structure which
can differ dependent on the envirionment the program runs in.
e.g. py,notebook,mlp,cluster""",
    )
    argparser.add_argument("--device", type=str, default="cuda", help="cuda/cpu")
    argparser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        help="should the program run in debug mode",
    )
    argparser.add_argument(
        "--project_path",
        type=str,
        help="the path to the project folder (otherwise the PF environment variable is used)",
    )
    argparser.add_argument(
        "--loch_path",
        type=str,
        help="the path to the loch folder (otherwise the LOCH environment variable is used)",
    )

    if arg_string:
        G.ARGS, G.ARGS_UNKNOWN = argparser.parse_known_args(shlex.split(arg_string))
    else:
        G.ARGS, G.ARGS_UNKNOWN = argparser.parse_known_args()

    G.DEBUG = G.ARGS.debug
    G.DOMAIN = G.ARGS.domain
    print(f"Startup with args: {G.ARGS} ", end="")


def load_device(device):
    """loads the device (cpu or cuda)"""

    global DEVICE

    DEVICE = "cpu"
    if device != "cpu":
        DEVICE = "cuda"


def load_env(env, device, job_id, project_path):
    """loads the environment from the yaml file

    :param env: id of the environment
    :param job_id: id of the job, necessary for SLURM cluster,
        if NONE, the job_id is not set
    ;param project_path: path to the project, overrides other paths
    """

    load_device(device)

    G.ENV = DD()
    G.PROJECT_ENV = DD()

    G.JOB["ID"] = None

    # load the paths
    if project_path is not None:
        G.ENV["PROJECT"] = project_path
    elif G.ARGS.project_path is not None:
        G.ENV["PROJECT"] = G.ARGS.project_path
    elif "PF" in os.environ:
        G.ENV["PROJECT"] = os.environ["PF"]
    else:
        G.ENV["PROJECT"] = os.getcwd()
    G.PROJECT_ENV["PROJECT"] = G.ENV.PROJECT

    if G.ARGS.loch_path is not None:
        print(f"Set Loch path to {G.ARGS.loch}")
        set_loch_path(G.ARGS.loch)

    print(f"ENV.PROJECT: '{G.ENV.PROJECT}'")
    yaml_env = file_to_str(os.path.join(G.ENV.PROJECT, "env", f"{env}.yaml"))
    yaml_env = yaml.load(yaml_env, Loader=yaml.FullLoader)

    # pylint: disable=unused-variable
    PF = G.ENV.PROJECT  # has to be here to be accessible by the eval function
    PATHS = DD()

    G.ENV["ID"] = eval(yaml_env["ID"])
    G.ENV["KIT"] = eval(yaml_env["KIT"])
    G.JOB["ID"] = get_next_job_id(env) if job_id == "AUTO" else job_id
    program_subfolders = yaml_env["PROGRAM_SUBFOLDERS"]

    for key, value in yaml_env["PATHS"].items():
        PATHS[key] = eval(value)
        if key in program_subfolders and G.PROGRAM is not None:
            G.PROJECT_ENV[key] = join(PATHS[key], G.PROJECT)
            G.ENV[key] = join(PATHS[key], G.PROGRAM)
        else:
            G.ENV[key] = G.PROJECT_ENV[key] = PATHS[key]

    G.ENV["JOBS"] = os.path.join(G.ENV.ARTEFACTS, "jobs")
    G.ENV["JOB"] = join(G.ENV.JOBS, G.JOB.ID) if G.JOB.ID else None

    # load the rest of the environment specific configurations
    for sub in ["CONFIGS"]:
        G.ENV[sub] = DD()
        for key, value in yaml_env[sub].items():
            G.ENV[sub][key] = eval(value)


def load_task(task, **kwargs):
    """loads the task from the yaml file

    :param task: task filename (without .yaml)
    :param kwargs: additional arguments to be passed to the DD.from_yaml function
    """

    G.TASK = DD()
    if task != "":
        task_file = os.path.join(G.ENV.CONFIG, "tasks", f"{task}.yaml")
        print(f"Load task file: {task_file}")
        G.TASK = DD.from_yaml(task_file, **kwargs)
    else:
        log_warning("TASK NOT SPECIFIED")


def load_mhparams(mhparams, **kwargs):
    """loads the model hyper-parameters (mhparams) from the yaml file

    :param mhparams: full mhparams filename including path
    :param kwargs: additional arguments to be passed to the DD.from_yaml function
    """

    if mhparams != "":
        G.MHPARAMS = G.MHPARAMS_STANDARD.update_from_yaml(mhparams, **kwargs)
    else:
        log_warning("MHPARAMS NOT SPECIFIED")


def load_dhparams(dhparams, **kwargs):
    """loads the data hyper-parameters (dhparams) from the yaml file

    :param dhparams: full dhparams filename including path
    :param kwargs: additional arguments to be passed to the DD.from_yaml function
    """

    if dhparams != "":
        G.DHPARAMS = DD.from_yaml(dhparams, **kwargs)
    else:
        log_warning("DHPARAMS NOT SPECIFIED")


def init(
    project,
    program,
    arg_string=None,
    logging_level=logging.INFO,
    project_path=None,
    create_job=True,
    docstring=None,
    argparser=None,
):
    """initializes the supporting kit framework which has the following structure

    a 'project' is the name of the project
    a 'task' is the task that the project's cape_run.py has to achieve (defined by a yaml file)
    a 'job' is the concrete execution of the task.
        Accomplishing a 'task' might require several 'jobs' (e.g. resume training a neural network)

    :param project: name of the project
    :param arg_string: string with arguments (project, env, task, job, debug)
    :param logging_level: logging level
    :param project_path: specify project path manually
    :param create_job: if True, a new job is created
    :param docstring: docstring of the main program.
    :param argparser: if not None, this argparser is used instead of the default one

    :return: args, args_unknown
    """

    global DEVICE, JOB_INFO
    JOB_INFO['time_init'] = now()

    if not G.MAIN_PROCESS:
        create_job = False

    set_args(arg_string, docstring, argparser)
    if G.DEBUG:
        logging_level = logging.DEBUG

    G.JOB = DD()
    G.MHPARAMS = DD()
    G.MHPARAMS_STANDARD = DD() if G.MHPARAMS_STANDARD is None else G.MHPARAMS_STANDARD
    G.DHPARAMS = DD()
    G.PROJECT = project
    G.PROGRAM = program

    # load environment
    load_env(
        G.ARGS.env, G.ARGS.device, G.ARGS.job if create_job else None, project_path
    )

    # setup the python logging system
    log_file = None
    G.CREATE_JOB = create_job
    if create_job:
        log_file = join(G.ENV.JOBS, G.JOB.ID, "job.log")
    elif not G.MAIN_PROCESS:
        logging_level = logging.ERROR
    setup_logger(logging_level, log_file)

    # load task definition
    load_task(G.ARGS.task)

    # Log summary to log files
    log_info(f"Start {G.PROJECT}-{G.TASK.ID}-{G.JOB.ID} on {G.ENV.ID} using {DEVICE}")

    log_info("G.ENV")
    for key, value in G.ENV.items():
        if key not in ["CONFIGS", "RUN"]:
            log_info(f"{key:<10s}... {value}")

    log_info("G.PROJECT_ENV")
    for key, value in G.PROJECT_ENV.items():
        if key not in ["CONFIGS", "RUN"]:
            log_info(f"{key:<10s}... {value}")

    log_info("Args: ")
    for arg, value in vars(G.ARGS).items():
        log_info(f"  {arg}={value} ")

    #
    # add logbook entry to task
    #
    if G.MAIN_PROCESS and G.TASK.ID is not None and G.JOB.ID is not None:
        str_to_file(
            f"{G.JOB.ID}\n",
            join(G.ENV.ARTEFACTS, "tasks", f"{G.TASK.ID}-{G.ENV.ID}.log"),
            append=True,
        )

    if create_job:
        d_args = vars(G.ARGS)
        d_args['args_unknown'] = G.ARGS_UNKNOWN
        DD.from_dict(d_args).to_yaml(join(G.ENV.JOB, "args.yaml"))

    return G.ARGS, G.ARGS_UNKNOWN


def shutdown():
    global JOB_INFO

    JOB_INFO['time_shutdown'] = now()

    if G.CREATE_JOB:
        JOB_INFO.to_yaml(join(G.ENV.JOB, "job_info.yaml"))
