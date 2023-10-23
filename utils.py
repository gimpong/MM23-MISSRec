import yaml
import re
import datetime


def build_yaml_loader():
    loader = yaml.FullLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(
            u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X
        ), list(u'-+0123456789.')
    )
    return loader


def load_config_files(file_list, yaml_loader):
    file_config_dict = dict()
    if file_list:
        for file in file_list:
            with open(file, 'r', encoding='utf-8') as f:
                file_config_dict.update(yaml.load(f.read(), Loader=yaml_loader))
    return file_config_dict


def get_local_time():
    r"""Get current time

    Returns:
        str: current time
    """
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y_%H-%M-%S')

    return cur


def chunked(lst, n):
    chunk_size = len(lst) // n
    chunks = []
    for i in range(n):
        if i == n - 1 and (i - 1) * chunk_size + chunk_size < len(lst):
            chunks.append(lst[i*chunk_size:])
        else:
            chunks.append(lst[i*chunk_size:i*chunk_size+chunk_size])

    return chunks