import logging
import numbers
from typing import Mapping, Any

from masuite import environments
from masuite.logging import base
from masuite.utils import wrappers

def wrap_environment(env: environments.Environment,
                     pretty_print: bool=True,
                     log_every: bool=False,
                     log_by_step: bool=False)->environments.Environment:
    logging.getLogger()
    logger = Logger(pretty_print)
    return wrappers.Logging(env, logger, log_by_step=log_by_step, log_every=log_every)


class Logger(base.Logger):
    def __init__(self, pretty_print: bool=True):
        self.pretty_print = pretty_print

    def write(self, data: Mapping[str, Any]):
        """Writes data to terminal"""
        if self.pretty_print:
            data = pretty_dict(data)

        print(data)


def pretty_dict(data: Mapping[str, Any])->str:
    msg = []
    for key in sorted(data):
        value = value_format(data[key])
        msg_pair = f'{key} = {value}'
        msg.append(msg_pair)

    return ' | '.join(msg)


def value_format(value: Any)->str:
    if isinstance(value, numbers.Integral):
        return str(value)
    if isinstance(value, numbers.Number):
        return f'{value:0.4f}'
    return str(value)