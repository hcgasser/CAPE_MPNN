import hashlib
import json


def dict_to_hash(dictionary, truncate=None):
    json_str = json.dumps(dictionary, sort_keys=True)
    return str_to_hash(json_str, truncate=truncate)


def str_to_hash(string, truncate=None):
    hash_key = hashlib.sha256(string.encode()).hexdigest()
    if truncate is not None:
        hash_key = hash_key[:truncate]
    return hash_key
