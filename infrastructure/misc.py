import os, sys
import pickle
import logging


def create_path(path): 
    try:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        #
        print("Directory '%s' created successfully" % (path))
    except OSError as error:
        print("Directory '%s' can not be created" % (path))
    #
    
def get_logger(logpath, displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)

    return logger
    
def cprint(color, text, **kwargs):
    if color[0] == '*':
        pre_code = '1;'
        color = color[1:]
    else:
        pre_code = ''
    code = {
        'a': '30',
        'r': '31',
        'g': '32',
        'y': '33',
        'b': '34',
        'p': '35',
        'c': '36',
        'w': '37'
    }
    print("\x1b[%s%sm%s\x1b[0m" % (pre_code, code[color], text), **kwargs)
    sys.stdout.flush()
    
    
# def save_object(obj, filename):
#     with open(filename, 'wb') as output:  
#         pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
        
# def load_object(filename):
#     with open(filename, 'rb') as input:
#         return pickle.load(input)