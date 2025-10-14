""" Global variables that should be accessible throughout the project. """

import os

from kit.time import now


#
# global variables
#
ARGS = None  # known arguments
ARGS_UNKNOWN = None  # unknown arguments
CREATE_JOB = False


DEBUG = False  # is debug mode active

PROJECT = ""  # project name
PROGRAM = ""  # program name
DOMAIN = ""  # the domain the code is running for (e.g. the protein)
ENV = None  # program environment variables, in particular paths to task, logs, MHPARAMS, etc.
PROJECT_ENV = (
    None  # project environment variables, in particular paths to DHPARAMS, etc.
)

TASK = None  # task definition
JOB = None  # job values like job id
MHPARAMS = None  # hyperparameters for the model
MHPARAMS_STANDARD = (
    None  # standard hyperparameters for the model that can be overwritten by yaml file
)

DHPARAMS = None  # hyperparameters for the data

START_TIME = now("%Y%m%d_%H%M")  # when the program was started

# determine rank
LOCAL_RANK = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0
GLOBAL_RANK = int(os.environ["RANK"]) if "RANK" in os.environ else 0
MAIN_PROCESS = GLOBAL_RANK == 0
