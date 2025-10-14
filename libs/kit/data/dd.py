""" the code for the DotDictionary class"""

import os
from collections import OrderedDict
import yaml

import kit.globals as G
from kit.utils import get_class
from kit.data.utils import file_to_str


class DD(OrderedDict):
    """DotDictionary

    implements the dot notation to access dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    @staticmethod
    def from_yaml(path, doc_idx=0, evaluate=True, **kwargs):
        """constructs a DD from a YAML file

        :param path: path to the YAML file
        :param doc_idx: index of the YAML document to use
        :param evaluate: whether to evaluate the values of the YAML document
        :param kwargs: additional arguments to pass to the evaluation
        :return: DD
        """

        x = file_to_str(path)
        x = list(yaml.load_all(x, Loader=yaml.FullLoader))
        x = DD.from_dict(x[doc_idx])
        if evaluate:
            x = x.evaluate(**kwargs)
        return x

    @staticmethod
    def from_dict(d):
        """constructs a DD from a dictionary
        supports nested dictionaries
        """

        result = DD()
        for key, value in d.items():
            if isinstance(value, dict):
                value = DD.from_dict(value)
            result[key] = value
        return result

    def update_from_yaml(self, path, doc_idx=0, evaluate=True, inplace=False, **kwargs):
        """updates a DD from a YAML file"""

        this = self if inplace else self.copy()
        x = DD.from_yaml(path, doc_idx=doc_idx, evaluate=False)
        for key, value in list(this.items()) + list(x.items()):
            this[key] = value
        if evaluate:
            this = this.evaluate(inplace=inplace, **kwargs)
        return this

    def to_dict(self):
        """converts a DD to a dictionary (including nested DDs)"""

        result = {}
        for key, value in self.items():
            if isinstance(value, DD):
                value = value.to_dict()
            result[key] = value
        return result

    def to_yaml(self, path, overwrite=False, encoding="utf-8"):
        """writes a DD to a YAML file"""

        if overwrite or not os.path.exists(path):
            this = self.make_evaluable().to_dict()
            with open(path, "w", encoding=encoding) as outfile:
                yaml.dump(this, outfile, default_flow_style=False)

    def make_evaluable(self, inplace=False):
        """encodes the DD as a string that can be evaluated to reconstruct the DD"""

        this = self if inplace else DD()
        for key, value in self.items():
            if isinstance(value, DD):
                this[key] = value.make_evaluable()
            else:
                if key == "CLS":
                    this[key] = f"{value.__module__}.{value.__name__}"
                elif isinstance(value, str):
                    this[key] = f'"{value}"'
                else:
                    this[key] = f"{value}"
        return this

    # pylint: disable=unused-argument
    def evaluate(self, inplace=False, **kwargs):
        """evaluates the values in the DD"""

        this = self if inplace else DD()
        for key, value in self.items():
            if isinstance(value, DD):
                this[key] = value.evaluate()
            else:
                if value == "INPUT" or value.startswith("OPTIONAL:"):
                    if key in G.ARGS or f"--{key}" in G.ARGS_UNKNOWN:
                        value = (
                            G.ARGS[key]
                            if key in G.ARGS
                            else G.ARGS_UNKNOWN[G.ARGS_UNKNOWN.index(f"--{key}") + 1]
                        )
                    elif value == "INPUT":
                        print(f"{key} (input gets evaluated): ", end="")
                        value = input()
                    else:
                        value = value.split(":")[1]

                if key == "CLS":
                    this[key] = get_class(value)
                else:
                    this[key] = eval(value)
        return this

    def copy(self):
        return DD.from_dict(super().copy())
