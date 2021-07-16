import inspect
import re
import time


def print_shape(*tensors):
    this_function_name = inspect.currentframe().f_code.co_name
    lcls = inspect.stack()
    outer = re.compile("\((.+)\)")
    arg_names = None
    for lcl in lcls:
        if lcl.code_context[0].strip().startswith(this_function_name):
            w = outer.search(lcl.code_context[0].split(this_function_name)[1])
            arg_names = w.group(1).split(", ")
    for idx, arg_name in enumerate(arg_names):
        print("# {0}.shape: {1}".format(arg_name, tensors[idx].shape))


# ex) yy-mm-dd/hh:mm:ss
def print_value(value):
    this_function_name = inspect.currentframe().f_code.co_name
    lcls = inspect.stack()
    outer = re.compile("\((.+)\)")
    arg_names = None
    for lcl in lcls:
        if lcl.code_context[0].strip().startswith(this_function_name):
            w = outer.search(lcl.code_context[0].split(this_function_name)[1])
            arg_names = w.group(1).split(", ")
    for idx, arg_name in enumerate(arg_names):
        print("# {0} : {1}".format(arg_name, value))


def current_time():
    return time.strftime('%y-%m-%d/%X', time.localtime(time.time()))
