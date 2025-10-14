""" This module contains functions for working with paths. """

import os
import re
from collections import defaultdict

from kit.log import log_info


def join(*args):
    """Joins the arguments to a path and creates the directory if it does not exist."""

    url = os.path.join(*args)
    
    if url.find(".") > 0:
        folder = os.path.sep.join(url.split(os.path.sep)[:-1] + [''])
    else:
        folder = url

    if folder == '':
        folder == '.'

    if not os.path.exists(folder):
        os.makedirs(folder)
        log_info(f"created directory: {folder}")
   
    return url


def get_entries(path, regex=None, returndict=True, subdirs=True):
    """Returns a list of entires (files and sub-directories)
    in a directory, optionally filtering by a regex.

    :param path: str - path to directory
    :param regex: str - regex to match the files/sub-directories
    :param returndict: bool - if True, returns a dict of lists,
        where the keys are the filenames without prefix and suffix

    :return result: list of files or dict of lists
    """

    result = defaultdict(lambda: []) if returndict else []
    pattern = re.compile(regex) if regex is not None else None
    if subdirs:
        for dirname, directories, filenames in os.walk(path):
            entries = directories + filenames
            for entry in entries:
                match = pattern.match(entry) if pattern is not None else True
                if match is not None:
                    l = result[entry] if returndict else result
                    l.append(os.path.join(dirname, entry))
    else:
        for entry in os.listdir(path):
            if pattern.match(entry):
                l = result[entry] if returndict else result
                l.append(os.path.join(path, entry))

    return result


def get_max_index(path, regex):
    """Returns the highest index of an entry in a directory,
    optionally filtered by prefix and suffix.

    :param path: str - path to directory
    :param regex: str - regex to match the files/sub-directories.
        The first group must be the index.

    :return max_index: int - highest index of a file/sub-directory
    """

    max_index = 0
    for _, dirnames, filenames in os.walk(path):
        entries = dirnames + filenames
        for entry in entries:
            match = re.match(regex, entry)
            if match is not None:
                index = int(match[1])
                if index > max_index:
                    max_index = index
    return max_index
