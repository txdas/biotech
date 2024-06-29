import os
import hashlib
import configparser
import warnings
import functools
import datetime
import logging
warnings.filterwarnings("ignore")
BASEDIR = os.path.abspath(os.path.dirname(__file__))
# CONFDIR = os.path.abspath(os.path.join(BASEDIR, "./conf/secrets.ini"))
# config = configparser.ConfigParser()
# config.read(CONFDIR)
# openai_api_key = config["OPENAI"]["OPENAI_API_KEY"]
# os.environ.update({"OPENAI_API_KEY": openai_api_key,
#                    "TOKENIZERS_PARALLELISM": "true"})
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(filename)s - %(funcName)s- %(lineno)d- '
                                               '%(thread)d- %(threadName)s  - %(message)s')


def relative_path(path):
    return os.path.abspath(os.path.join(BASEDIR,path))



def count(fn):
    ret = os.popen("wc -l %s | awk '{print $1 }'" % fn)
    return int(ret.read().strip())


def mhash(s, th=1000):
    md5 = hashlib.md5()
    md5.update(s.encode('utf-8'))
    value = md5.hexdigest()
    s = 0
    for v in range(0, len(value), 4):
        s += int(value[v:v + 4], 16) % th
    return s % th


def batch(generator, batch_size=1000):
    assert (isinstance(batch_size, int) and batch_size > 0)
    lst = []
    for i, v in enumerate(generator):
        lst.append(v)
        if (i + 1) % batch_size == 0:
            yield lst
            lst = []
    if lst:
        yield lst


def except_wrap(fun):
    @functools.wraps(fun)
    def wrapper(*args, **kwds):
        try:
            value = fun(*args, **kwds)
            return value
        except:
            import traceback
            traceback.print_exc()
            pass
    return wrapper


def time_wrap(fun):
    @functools.wraps(fun)
    def wrapper(*args, **kwds):
        logging.info("*********\t method %s is invoked. \t ************* \n" % fun.__name__)
        start_time = datetime.datetime.now()
        value = fun(*args, **kwds)
        end_time = datetime.datetime.now()
        time_diff = end_time - start_time
        logging.info("method %s is completed and costs about %s. \n  " % (fun.__name__, str(time_diff)))
        return value

    return wrapper


if __name__ == '__main__':
    for v in batch(range(100)):
        print(v)
