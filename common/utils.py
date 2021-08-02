import inspect
import re
import time


def print_shape(*tensors):
    this_function_name = inspect.currentframe().f_code.co_name
    lcls = inspect.stack()
    outer = re.compile("\((.+)\)")
    arg_names = None
    for lcl in lcls:
        if this_function_name in lcl.code_context[0].strip():
            w = outer.search(lcl.code_context[0].split(this_function_name)[1])
            arg_names = w.group(1).split(", ")
    for idx, arg_name in enumerate(arg_names):
        print("# {0}.shape: {1}".format(arg_name, tensors[idx].shape))


def printv(value):
    this_function_name = inspect.currentframe().f_code.co_name
    lcls = inspect.stack()
    outer = re.compile("\((.+)\)")
    arg_names = None
    for lcl in lcls:
        if this_function_name in lcl.code_context[0].strip():
            w = outer.search(lcl.code_context[0].split(this_function_name)[1])
            arg_names = w.group(1).split(", ")
    for idx, arg_name in enumerate(arg_names):
        print("# {0} : {1}".format(arg_name, value))


# ex) yy-mm-dd/hh:mm:ss
def current_time():
    return time.strftime('%y-%m-%d/%X', time.localtime(time.time()))


def overview_env(env):
    printv(env.action_space)
    printv(env.get_action_meanings())
    printv(env.observation_space.shape)
