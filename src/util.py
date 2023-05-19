import os
import re
from typing import Any, Dict, Iterator, List, Mapping, Tuple

rtx2080ti = 11020
v100 = 16400  # estimate
a6000 = 49000  # estimate

HUGGINGFACE_CACHE_DIR = "cache/huggingface"


def sizeof_fmt(num, suffix="B"):
    # https://stackoverflow.com/a/1094933/14390930
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


def send_to_device(obj, device):
    if isinstance(obj, list):
        return [send_to_device(t, device) for t in obj]

    if isinstance(obj, tuple):
        return tuple(send_to_device(t, device) for t in obj)

    if isinstance(obj, dict):
        return {
            send_to_device(key, device): send_to_device(value, device)
            for key, value in obj.items()
        }

    if hasattr(obj, "to"):
        return obj.to(device)

    return obj


def flattened(obj: object) -> Iterator[Any]:
    """
    Given a list, dictionary, or other value, yields an iterator of dictionaries/other values with no lists anywhere inside the structure.
    """
    if not isinstance(obj, dict) and not isinstance(obj, list):
        yield obj
        return

    if isinstance(obj, list):
        for elem in obj:
            yield from flattened(elem)
        return

    assert isinstance(obj, dict)

    if not obj:
        yield obj
        return

    flat_list = {field: list(flattened(value)) for field, value in obj.items()}

    yield from _flatten_dict_of_lists(flat_list)


def _flatten_dict_of_lists(
    dict_of_lists: Mapping[object, List[object]],
) -> Iterator[Dict[object, object]]:
    assert isinstance(dict_of_lists, dict)

    if not dict_of_lists:
        yield dict_of_lists
        return

    field, value_list = dict_of_lists.popitem()

    assert isinstance(value_list, list)

    for value in value_list:
        for flattened_dict in _flatten_dict_of_lists(dict_of_lists):
            yield {**flattened_dict, field: value}

    dict_of_lists[field] = value_list


def fields_with_lists(obj: object) -> Iterator[Tuple[str, ...]]:
    """
    In a nested dictionary, returns a generator of any paths to fields that contain a list.
    """

    if not isinstance(obj, dict) and not isinstance(obj, list):
        return

    if isinstance(obj, list):
        for elem in obj:
            yield from fields_with_lists(elem)
        return

    assert isinstance(obj, dict)

    if not obj:
        return

    for field, value in obj.items():
        if isinstance(value, list):
            yield (field,)

        for nested_field in fields_with_lists(value):
            yield (field, *nested_field)


def index_dict(dct, path):
    path = list(reversed(path))

    while path:
        dct = dct[path.pop()]

    return dct


def files_with_match(patterns: List[str]) -> Iterator[str]:
    for dirpath, _, filenames in os.walk("."):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            for pattern in patterns:
                if re.search(pattern, filepath):
                    yield filepath


def files_with_extension(paths: List[str], ext: str) -> Iterator[str]:
    # add . to ext if it doesn't have it.
    ext = "." + ext if ext[0] != "." else ext

    for path in paths:
        if os.path.isfile(path):
            if path.endswith(ext):
                yield path
                continue

        for dirpath, _, filenames in os.walk(path):
            for filename in filenames:
                if filename.endswith(ext):
                    yield os.path.join(dirpath, filename)
