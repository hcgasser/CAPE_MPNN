""" This module contains auxiliary functions for data processing. """

import os
from enum import IntEnum
import re

import numpy as np


class Split(IntEnum):
    """enum for the different types of splits in a ML project"""

    TRAIN = 0
    VAL = 1
    TEST = 2
    PREDICT = 3

    def __str__(self):
        """return the string name of the split (TRAIN/VAL/TEST/PREDICT)"""
        return super().__str__().split(".")[-1]

    def long_name(self):
        if self == Split.TRAIN:
            return "training"
        if self == Split.VAL:
            return "validation"
        if self == Split.TEST:
            return "testing"
        if self == Split.PREDICT:
            return "prediction"
        raise Exception(f"unknown split {self}")

    @staticmethod
    def get_palette():
        """return the seaborn color palette linked with the splits"""
        return {s.long_name(): f"C{int(s)*2}" for s in Split}


def str_to_file(s, filename, append=False, encoding="utf-8"):
    """writes a string to a file"""

    mode = "a" if append else "w"
    with open(filename, mode, encoding=encoding) as file:
        print(s, end="", file=file)


def file_to_str(filename, encoding="utf-8"):
    """reads a file and returns its content as a string"""

    if not os.path.exists(filename):
        return ""

    with open(filename, "r", encoding=encoding) as file:
        return file.read()


def scale_df(df):
    """scales a dataframe to the range [-1, 1]"""

    maximum = df.max().max()
    minimum = df.min().min()

    scale = max(maximum, -minimum)

    return df / scale


def set_df_cell_to_np(df, idx, column, array, dtype=np.dtype("float32")):
    """sets a dataframe cell to a numpy array"""
    df.at[idx, column] = array.astype(dtype).tobytes()


def get_np_from_df_cell(df, idx, column, shape=None, dtype=np.dtype("float32")):
    """returns a numpy array from a dataframe cell"""
    value = np.frombuffer(df[idx, column], dtype=dtype)
    if shape is not None:
        value = value.reshape(shape)
    return value


def remove_lines(lines_or_file_path, regex):
    """removes all lines that match a certain regex"""
    input_is_file = False
    if isinstance(lines_or_file_path, list):
        input_lines = lines_or_file_path
    elif isinstance(lines_or_file_path, str) and os.path.exists(lines_or_file_path):
        input_is_file = True
        input_lines = file_to_str(lines_or_file_path).split("\n")
    else:
        raise Exception(
            f"lines must be a list or a file path, but is {lines_or_file_path}"
        )
    output_lines = []

    if isinstance(regex, str):
        regex = [regex]

    for line in input_lines:
        if not any(re.match(r, line) is not None for r in regex):
            output_lines.append(line)

    if input_is_file:
        str_to_file("\n".join(output_lines), lines_or_file_path)

    return output_lines


def remove_lines_between(
    lines_or_file_path, re_start, re_end=None, remove_start=True, remove_end=True
):
    """removes all lines between two lines that match two regexes"""
    input_is_file = False
    if isinstance(lines_or_file_path, list):
        input_lines = lines_or_file_path
    elif isinstance(lines_or_file_path, str) and os.path.exists(lines_or_file_path):
        input_is_file = True
        input_lines = file_to_str(lines_or_file_path).split("\n")
    else:
        raise Exception(
            f"lines must be a list or a file path, but is {lines_or_file_path}"
        )
    output_lines = []

    output, switch_next = True, False
    for line in input_lines:
        if switch_next:
            output, switch_next = not output, False

        if re_start is not None and re.match(re_start, line):
            if output:
                if remove_start:
                    output = False
                else:
                    switch_next = True

        if re_end is not None and re.match(re_end, line):
            if not output:
                if remove_end:
                    switch_next = True
                else:
                    output = True
        if output:
            output_lines.append(line)

    if input_is_file:
        str_to_file("\n".join(output_lines), lines_or_file_path)

    return output_lines


def df_get_highest_in_category(df, tgt_column, category_column, query):
    """returns the highest value in a certain column (tgt_column)
    for each value in another column (category_column)"""

    df = df.reset_index()
    df = df.query(query)
    df = df[[tgt_column, category_column]].groupby(category_column).max()
    return df


def invert_dict(d):
    """inverts a dictionary

    the values of the input dictionary become the keys of the output dictionary
    the keys of the input dictionary become the values in a list of the output dictionary
    
    :param d: dictionary to invert
    :return: inverted dictionary
    """
    result = {}
    for k, v in d.items():
        if v not in result:
            result[v] = []
        result[v].append(k)
    return result