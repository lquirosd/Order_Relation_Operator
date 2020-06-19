import os
import errno


def mkdir(path):
    try:
        os.makedirs(os.path.expanduser(os.path.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and os.path.isdir(path):
            raise e


def files_exist(files):
    return all([os.path.exists(f) for f in files])
