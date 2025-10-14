""" Utility functions for the kit library """

import os
import importlib
import tempfile
from contextlib import contextmanager
import json
import hashlib


def get_class(classname):
    """returns the class object for the given class name

    :param classname: name of the class
    :return: class object
    """

    parts = classname.split(".")
    if len(parts) > 1:
        importlib.import_module(parts[0])
        module = ".".join(parts[:-1])
        module = importlib.import_module(module)
        classname = f"module.{parts[-1]}"

    return eval(classname)


@contextmanager
def temp_working_directory(dir_path=None):
    """Context manager for a temporary working directory

    Will change the current working directory to a temporary directory
    if no directory path is given. Otherwise, it will change the current
    working directory to the given directory path.

    When the context manager exits, the current working directory is
    changed back to the original working directory.

    :param dir_path: path to the directory to change to
    """

    original_cwd_path = os.getcwd()
    try:
        if dir_path is None:
            with tempfile.TemporaryDirectory() as tmp_dir_path:
                os.chdir(tmp_dir_path)
                yield tmp_dir_path
        else:
            os.chdir(dir_path)
            yield dir_path

    finally:
        os.chdir(original_cwd_path)


def hash_object(obj):
    """Computes a hash for an object"""
    
    serialized_data = json.dumps(obj, sort_keys=True, default=str) 
    return hashlib.sha256(serialized_data.encode()).hexdigest()

