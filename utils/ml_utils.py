"""
A collection of general purpose utility procedures for machine-learning.
NOTE: Most procedures here are NOT used in this project
Code by Yuval Atzmon: : https://github.com/yuvalatzmon/LAGO
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import random
import re
import collections
import subprocess
import time
import os

import pickle
from contextlib import contextmanager
import json
from math import log10, floor  # for round_sig
import sys
import glob
import argparse

import numpy as np
import pandas as pd


def get_latest_file(fullpath):
    """Returns the name of the latest (most recent) file
    of the joined path(s)
    Modified from: https://codereview.stackexchange.com/a/120500
    """
    list_of_files = glob.glob(fullpath + '/*')  # You may use iglob in Python3
    if not list_of_files:  # I prefer using the negation
        return None  # because it behaves like a shortcut
    latest_full_file = max(list_of_files, key=os.path.getctime)
    return latest_full_file


def last_modification_time(full_path_name):
    file_mod_time = os.stat(full_path_name).st_mtime
    # Time in minutes since last modification of file
    return (time.time() - file_mod_time) / 60.


def safe_path_delete(path_to_del, base_path):
    """ Raise an error if path_to_del is not under base_path"""
    path_to_del = os.path.expanduser(path_to_del)
    if path_to_del.startswith(os.path.expanduser(base_path)):
        print('Deleting %s' % path_to_del)
        print(run_bash('rm -r %s' % path_to_del))
    else:
        raise ValueError('path_to_del %s is not under base path %s' % (
            path_to_del, base_path))


def safe_path_content_delete(path_to_del, base_path, except_list):
    """ Delete the content of a path.
        Raise an error if path_to_del is not under base_path"""
    path_to_del = os.path.expanduser(path_to_del)
    if path_to_del.startswith(os.path.expanduser(base_path)):

        if except_list:
            except_str = ' '.join([f'"{os.path.basename(fname)}"' for fname in except_list])
            print(f'Deleting {path_to_del}/*, except {except_str}')
            print(run_bash(f'find {path_to_del} -type f -not -name {except_str} -delete'))
        else:
            print('Deleting %s/*' % path_to_del)
            print(run_bash('rm -r %s/*' % path_to_del))
    else:
        raise ValueError('path_to_del %s is not under base path %s' % (
            path_to_del, base_path))


def path_exists_with_wildcard(full_path_filename, depth=1):
    dirname = os.path.dirname(os.path.expanduser(full_path_filename))
    filename = os.path.basename(os.path.expanduser(full_path_filename))

    print(f'find {dirname} -maxdepth {depth} -name {filename} ')
    find_results = run_bash(f'find {dirname} -maxdepth {depth} -name {filename} ')
    path_exists = len(find_results) > 0

    return path_exists


def check_experiment_progress(full_path_name, results_fname, touch_fname,
                              in_progress_timeout_minutes,
                              obsolete_result_timeout_hours=None,
                              base_output_dir=None):
    """ Check the filesystem for experiment progress

    check_experiment_progress() checks if the current selection of
    params is already In-progress or Completed. If True, then exit, and don't train the model.
    This allows to execute **in parallel**, without conflicts, several
    hyper-param search scripts that use the same filesystem (on multiple
    machines or GPUs).

    Note: This was not exhaustively tested to avoid conflict 100%of the times,
    but we are OK if on rare times, two training scripts are executed with
    the same parameters. The only problem in such case is that the
    training_log csv file may become corrupt.

    The general idea, is that we give a meaningful name to the train_dir, such
    that it describes the configuration of the current experiment.
    Then, on every epoch a file name is "touched" on the train_dir. This file
    indicates that training is in progress in this directory, and therefore
    we will exit this instance.
    Moreover, if the results_fname file exists, it means that this experiment
    had completed. In that case, we also exit this instance.
    There are two exceptions for the above conditions:
    (1) if obsolete_result_timeout_hours exceeds, then we rerun this instance.
    (2) if in_progress_timeout_minutes exceeds, then we will keep running
        this instance.

    """

    # introduce some start time variability, to reduce conflicts
    sleeptime = random.random() * 2
    time.sleep(sleeptime)

    full_path_name = os.path.expanduser(full_path_name)
    if not results_fname.startswith(full_path_name):
        results_fname = os.path.join(full_path_name, results_fname)
    if not touch_fname.startswith(full_path_name):
        touch_fname = os.path.join(full_path_name, touch_fname)

    def _get_state():
        state = 'Unfulfilled'
        if os.path.exists(full_path_name):
            print('DEBUG: check_experiment_progress: exists(full_path_name)')
            # Setting status indicators:
            result_is_obsolete = False
            check_obsolete = obsolete_result_timeout_hours is not None
            if check_obsolete and os.path.exists(results_fname):
                dt_obsolete = last_modification_time(results_fname) / 60.
                if dt_obsolete > obsolete_result_timeout_hours:
                    result_is_obsolete = True
            recent_touch = False
            if os.path.exists(touch_fname):
                dt_touch = last_modification_time(touch_fname)
                if dt_touch < in_progress_timeout_minutes:
                    recent_touch = True

            # Checking experiment state:
            if os.path.exists(results_fname):
                state = 'Completed'
                print('DEBUG: exists(results_fname)')
                if check_obsolete and result_is_obsolete:
                    state = 'Unfulfilled'
                    print('DEBUG: check_obsolete and result_is_obsolete')
                    if recent_touch:
                        state = 'In-Progress'
                        print('DEBUG: recent_touch')
            else:
                if recent_touch:
                    print('DEBUG: recent_touch')
                    state = 'In-Progress'
        return state

    state = _get_state()
    if state == 'Unfulfilled':
        try:
            run_bash(f'source src/utils/atomic_touch.sh {touch_fname}',
                     versbose=False)
        except RuntimeError as err:
            if 'cannot overwrite existing file' in str(err):
                state = 'In-Progress'
            else:
                raise RuntimeError(err)
    if state == 'Unfulfilled':
        # Delete the results under the obsolete / unfulfilled path,
        # if it is not empty
        print('DEBUG: "Deleting files (if exist) under the '
              'unfulfilled/obsolete path: %s"' % full_path_name)
        if base_output_dir is not None:  # for backward compatibility
            # check if path is not empty
            if len(list(os.walk(full_path_name))[0][2]) > 0:
                safe_path_content_delete(full_path_name, base_output_dir,
                                         except_list=[touch_fname])
        print('DEBUG: Unfulfilled')

    print('DEBUG: final state = %s' % state)
    return state


def touch(full_name):
    # Python 3
    from pathlib import Path

    Path(full_name).touch()


def path_reduce_user(pathname, begins_with_path=True):
    """The opposite of os.path.expanduser()
        Reduces full path names to ~/... when possible """
    if begins_with_path:
        if pathname.startswith(os.path.expanduser('~')):
            pathname = pathname.replace(os.path.expanduser('~'), '~', 1)
    else:
        pathname = pathname.replace(os.path.expanduser('~'), '~')

    return pathname


def get_current_git_hash(dir='./', shorten=True):
    """ Returns a (hex) string with the current git commit hash.
    """
    short = ''
    if shorten:
        short = '--short'

    dir = os.path.expanduser(dir)
    git_hash = run_bash('cd %s && git rev-parse %s HEAD' % (dir, short))
    return git_hash


def get_username():
    """ 
    :return: username by last part of os.path.expanduser('~')
    """""
    return os.path.basename(os.path.expanduser('~'))


def slice_dict_to_tuple(d, keys):
    """ Returns a tuple from dictionary values, ordered and slice by given keys
        keys can be a list, or a CSV string
    """
    if isinstance(keys, str):
        keys = keys[:-1] if keys[-1] == ',' else keys
        keys = re.split(', |[, ]', keys)

    return [d.get(k, None) for k in keys]


def replace_filename_in_fullname(fullname, new_name):
    return os.path.join(os.path.dirname(fullname), new_name)


def replace_file_extension(fname, new_ext):
    """ replace filename extension """
    if new_ext[0] != '.':
        new_ext = '.' + new_ext
    return os.path.splitext(fname)[0] + new_ext


def show_redundant_command_line_args():
    import tensorflow as tf

    FLAGS = tf.flags.FLAGS
    _, parsed_flags = argparse.ArgumentParser().parse_known_args()
    redundant_args = set(
        map(lambda x: re.split('[ =]', x[2:])[0], parsed_flags)) - set(
        vars(FLAGS)['__flags'])
    redundant_args = list(redundant_args)

    msg = 'Passed redundant command line arguments: {}'.format(redundant_args)
    eprint(msg)
    tf.logging.info(msg)


def normalize_data_per_sample(DATA_samples_x_features):
    """ Normalize each sample to zero mean and unit variance
        input data dimensions is samples_x_features
    """
    X = DATA_samples_x_features
    X_zero_mean = ((X.T - X.mean(axis=1)).T)
    X_L2_normed = ((X_zero_mean.T / np.linalg.norm(X_zero_mean, axis=1)).T)
    return X_L2_normed


def join_strings_left_right_align(lhs_str, rhs_str, min_str_length):
    output = lhs_str + ' ' * (min_str_length - len(lhs_str) - len(rhs_str))
    output += rhs_str
    return output


def get_gpus_stat():
    try:
        gpus_stat = run_bash('utils_src/gpustat.py --no-color').split('\n')[1:]
    except RuntimeError:
        gpus_stat = []
    return gpus_stat


def all_gpus_ids():
    return list(range(len(get_gpus_stat())))


def find_available_gpus(mem_threshold=1000):
    """ Iterate on stats per gpu. Return only GPUs that
        used memory is below threshold

        THIS PROCEDURE IS BUGGY when multiple instantiations run in parallel.
        To fix add mutex mechanism
        """

    # Call a script that nicely parse nvidia-smi to GPUs statistics
    gpus_stat = get_gpus_stat()
    gpus_list = []
    # Iterate on stats per gpu.
    for gpu_id, gpu in enumerate(gpus_stat):
        used_mem = int(re.findall('(\d+) / (\d+) MB', gpu)[0][0])
        # Take only GPUs that used memory is below threshold
        print('used_mem=', used_mem)
        if used_mem < mem_threshold:
            gpus_list.append(gpu_id)

    if len(gpus_list) == 0 and len(gpus_stat) > 0:
        raise RuntimeError('No GPUs are available\n'
                           'Uses memory threshold=%d\n'
                           'GPUs stats:\n%s' % (mem_threshold, '\n'.join(gpus_stat)))
    return gpus_list


def sigmoid(x):
    """ Numerically stable numpy sigmoid
        https://stackoverflow.com/a/29863846/2476373
    """
    return np.exp(-np.logaddexp(0, -np.array(x)))


def speedup_keras_loss(loss_list, output_layers_list, loss_weights):
    """ Speedup keras training by nulling loss tensors where loss weight=0
    and setting the output layers to be non-trainable"""
    from keras import backend as K

    def null_loss(y_true, y_pred):
        return K.constant(0.)

    for n, weight in enumerate(loss_weights):
        if weight == 0:
            output_layers_list[n].trainable = False
            loss_list[n] = null_loss
    return loss_list


def unique_rows_np(a):
    # http://stackoverflow.com/a/31097277
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)] * a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))


def is_PID_running(PID, hostname=None):
    # Adpated from http://stackoverflow.com/a/15774758/2476373

    if hostname is None:
        pid_cmd = """ps -p {PID}""".format(PID=PID)
    else:
        pid_cmd = """ssh {hostname} "ps -p {PID}" """.format(PID=PID,
                                                             hostname=hostname)

    return len(run_bash(pid_cmd).split('\n')) > 1


def eprint(*args, **kwargs):
    """ Print to stderr
        http://stackoverflow.com/a/14981125/2476373
    """
    print(*args, file=sys.stderr, **kwargs)


def write_hostname_pid_to_filesystem(filename):
    my_mkdir(os.path.dirname(filename))

    with open(filename, 'w') as f:
        json.dump(dict(hostname=run_bash('hostname'), pid=os.getpid()), f,
                  indent=4)


def read_hostname_pid_from_filesystem(filename):
    with open(filename, 'r') as f:
        d = json.load(f)
    return d['hostname'], d['pid']


def grep(s, pattern):
    """ grep
    """
    # Adapted from http://stackoverflow.com/a/25181706
    return '\n'.join(
        [line for line in s.split('\n') if re.search(pattern, line)])


def data_to_pkl(data, fname):
    # pickle data
    with open(fname, 'wb') as f:
        pickle.dump(data, f)


def dict_to_arg_flags_str(flags_dict):
    """
    Converts a dictionary to a commandline arguments string
    in the format '--<key0>=value0 --<key1>=value1 ...'
    """
    return ' '.join(
        ['--{}={}'.format(k, flags_dict[k]) for k in flags_dict.keys()])


def tensorflow_flags_to_str(tf_flags):
    """ Convert tensorflow FLAGS to a commandline arguments string
    in the format '--<flag0>=value0 --<flag1>=value1 ...' """
    flags_dict = vars(tf_flags)['__flags']
    return dict_to_arg_flags_str(flags_dict)


def generate_split_indices(samples_ids, seed, split_ratios):
    """
    Generate indices for a cross validation (XV) split, given ids of samples, or just number of samples

    :param samples_ids:  given ids of samples (list or ndarray), or a scalar indicating number of samples
    :param seed: random seed
    :param split_ratios: a list of ratios for XV sets
    :return: a list of lists of indices per XV set

    Example:
        train_ids, val_ids, test_ids = generate_split_indices(Nimages, seed=111, split_ratios=[0.6, 0.2, 0.2])

    """
    samples_ids = np.array([samples_ids]).astype(int).flatten()

    if samples_ids.size == 1:
        samples_ids = np.array(range(samples_ids[0])).astype(int).flatten()

    nsamples = samples_ids.size

    # Set a seed for the split, with a temporary context
    with temporary_random_seed(seed):
        # Calc number of samples per set
        n_indices = np.round(nsamples * np.array(split_ratios)).astype(int)
        n_indices[-1] = nsamples - np.sum(n_indices[0:-1])

        # Draw a random permutation of the samples IX
        perm = np.random.permutation(nsamples)
        # Split the random permutation IX sequentially according to num of samples per set
        ix0 = 0
        ix_set = []
        for N_ix in n_indices.tolist():
            ix1 = ix0 + int(N_ix)
            ix_set += [perm[ix0:ix1].tolist()]
            ix0 = ix1

    # Assign ids according to the split IX
    ids_set = []
    for ix in ix_set:
        ids_set += [np.take(samples_ids, ix).tolist()]

    return ids_set


def my_mkdir(dir_name):
    # mkdir if not exist
    return os.makedirs(dir_name, exist_ok=True)


def load_dict(fname, var_names, load_func=pickle.load):
    """ Loads specific keys from a dictionary that was to a file
    :type fname: file name
    :type var_names: variables to retrieve. Can be a list or comma seperated string
          e.g. 'a, b,c' or ['a', 'b', 'c']
    :param load_func: default: pickle.load
    """
    if type(var_names) == str:
        var_names = re.split(', ?[, ]?', var_names)
    with open(fname, "rb") as f:
        data_dict = load_func(f)
    assert isinstance(data_dict, dict)
    return tuple([data_dict[var] for var in var_names])


def cond_load_dict(fname, var_names, do_force=False, load_func=pickle.load):
    """
    usage:
    data_dict, do_stage = cond_load_dict(fname, 'x,y,z', do_force):
       if do_stage:
           data_dict = <calculate the data>
           <save the data to fname>

       return data_dict, do_stage
    """
    do_stage = True
    if type(var_names) == str:
        var_names = re.split(', ?[, ]?', var_names)

    # Python 2 to 3 compatibility
    if sys.version_info[0] > 2 and load_func == pickle.load:
        def load_func23(f):
            return load_func(f, encoding='latin1')
    else:
        load_func23 = load_func

    # noinspection PyBroadException
    try:
        with open(fname, "rb") as f:
            data_dict = load_func23(f)
        assert isinstance(data_dict, dict)

        # check if all required var names are members of loaded data
        if np.in1d(var_names, list(data_dict.keys())).all():
            do_stage = False
    except:
        do_stage = True

    if do_stage or do_force:
        return tuple([True] + [None for _ in var_names])
    else:
        # noinspection PyUnboundLocalVariable
        return tuple([False] + [data_dict[var] for var in var_names])


@contextmanager
def temporary_random_seed(seed):
    """ A context manager for a temporary random seed (only within context)
        When leaving the context the numpy random state is restored
        Inspired by http://stackoverflow.com/q/32679403
    """
    state = np.random.get_state()
    np.random.seed(seed)
    yield None
    np.random.set_state(state)


def remove_dict_key(d, key):
    """remove_dict_key(d, key)
       Removes a key-value from a dict.
       Do nothing if key does not exist
       :param d: dictionary
       :type key:
       """

    if key in d:
        d.pop(key)

    return d


# noinspection PyPep8Naming
def ismatrix(M):
    if len(M.shape) == 1 or any(np.array(M.shape) == 1):
        return False
    else:
        return True


# noinspection PyPep8Naming
def get_prec_at_k(a_ind):
    # evaluate 'precision at k' for all k's (number of positive indications (of higest k scores) / k)
    """

    :type a_ind: matrix
    """
    assert (ismatrix(a_ind))

    K, N = a_ind.shape
    a_ind = a_ind.astype('float32')

    precision_at_all_ks = (a_ind.cumsum(0).T / range(1, K + 1)).mean(
        0)  # used .T for broadcasting
    # alternative: a_ind.sum(1).cumsum()/range(1,K+1)/N

    return precision_at_all_ks


def run_bash(cmd, raise_on_err=True, raise_on_warning=False, versbose=True):
    """ This function takes Bash commands and return their stdout
    Returns: string (stdout)
    :type cmd: string
    """
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, executable='/bin/bash')
    # p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    out = p.stdout.read().strip().decode('utf-8')
    err = p.stderr.read().strip().decode('utf-8')
    if err and raise_on_err:
        do_raise = True
        if 'warning' in err.lower():
            do_raise = raise_on_warning
            if versbose:
                print('command was: {}'.format(cmd))
            eprint(err)

        if do_raise:
            if versbose:
                print('command was: {}'.format(cmd))
            raise RuntimeError(err)

    return out  # This is the stdout from the shell command


def build_string_from_dict(d, sep='__'):
    """
     Builds a string from a dictionary.
     Mainly used for formatting hyper-params to file names.
     Key-Value(s) are sorted by the key, and dictionaries with
     nested structure are flattened.

    Args:
        d: dictionary

    Returns: string
    :param d: input dictionary
    :param sep:

    """
    fd = _flatten_dict(d)
    return sep.join(
        ['{}={}'.format(k, _value2str(fd[k])) for k in sorted(fd.keys())])


def slice_dict(d, keys_list):
    return {k: v for k, v in d.items() if k in keys_list}


def grouped(iterable, n, incomplete_tuple_ok=True):
    """ http://stackoverflow.com/a/5389547/2476373and http://stackoverflow.com/a/38059462
    s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), (s2n,s2n+1,s2n+2,...s3n-1), ...\

    Usage example:
    for x, y in grouped(range(10:21), 2):
        print "%d + %d = %d" % (x, y, x + y)

    """

    if incomplete_tuple_ok:
        def zip_discard_compr(*iterables):
            """izip_longest which discards missing values and produce incomplete tuples"""
            sentinel = object()
            return [[entry for entry in iterable if entry is not sentinel]
                    for iterable in
                    itertools.izip_longest(*iterables, fillvalue=sentinel)]

        return zip_discard_compr(*[iter(iterable)] * n)
    else:
        return itertools.izip(*[iter(iterable)] * n)


def join_path_with_extension(path_parts, extension=None):
    """ Join path parts and safely adding extension

    path_parts: list of parts of path
    extension: file extension, if set to None, just calls os.path.join(*path_parts)

    returns full path with extension

    Examples:

    >>> join_path_with_extension(['a', 'b', 'c'], 'jpg')
    'a/b/c.jpg'
    >>> join_path_with_extension(['a', 'b', 'c'], '.jpg')
    'a/b/c.jpg'
    >>> join_path_with_extension(['a', 'b', 'c.jpg'], 'jpg')
    'a/b/c.jpg'
    >>> join_path_with_extension(['a', 'b', 'c.jpg'], '.jpg')
    'a/b/c.jpg'
    >>> join_path_with_extension(['a', 'b', 'cjpg'], '.jpg')
    'a/b/cjpg.jpg'
    >>> join_path_with_extension(['a', 'b', 'c'])
    'a/b/c'
    """
    full_path = os.path.join(*path_parts)

    if extension is not None:
        if extension[0] != '.':
            extension = '.' + extension

        if not full_path.endswith(extension):
            full_path += extension

    return full_path


def round_sig(x, sig=2):
    # http://stackoverflow.com/a/3413529/2476373
    if x == 0:
        return 0
    else:
        return round(x, sig - int(floor(log10(abs(x)))) - 1)


def dataframe_from_json(fname, columns):
    if os.path.exists(fname):
        df = pd.read_json(fname)
    else:
        df = pd.DataFrame(None, columns=columns)

    return df


def dataframe_from_csv(fname, columns, sep=','):
    if os.path.exists(fname):
        df = pd.read_csv(fname, sep=sep)
    else:
        df = pd.DataFrame(None, columns=columns)

    return df


def read_json(fname, return_None_if_not_exist=True):
    if os.path.exists(fname):
        with open(fname, 'r') as f:
            data = json.load(f)

    else:
        if not return_None_if_not_exist:
            raise Exception('Error, no such file: %s' % fname)
        else:
            data = None
    return data


def merge_dict_list(dict_list):
    # http://stackoverflow.com/a/3495415/2476373
    return dict(kv for d in dict_list for kv in d.iteritems())


def read_modify_write_json(fname, update_dict, create_if_not_exist=True):
    """
    Read-Modify-Write a JSON file
    NOTE that it only modify on 1 level. Nested dicts are over written.
    Args:
      fname: filename
      update_dict: update values
      create_if_not_exist: default = True

    Returns:

    """
    if os.path.exists(fname):
        with open(fname, 'r') as f:
            data = json.load(f)
    else:
        if not create_if_not_exist:
            raise Exception('Error, no such file: %s' % fname)
        else:
            data = {}
            with open(fname, 'w') as f:
                f.write(json.dumps(data, indent=4))

    data.update(update_dict)
    with open(fname, 'w') as f:
        f.write(json.dumps(data, indent=4))


# Homemade version of matlab tic and toc functions
# from http://stackoverflow.com/a/18903019/2476373
def tic():
    # noinspection PyGlobalUndefined
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()


def toc():
    if 'startTime_for_tictoc' in globals():
        print("Elapsed time is " + str(
            time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print("Toc: start time not set")


# --- Auxilary functions ---


def _flatten_dict(d, parent_key='', sep='_'):
    # from http://stackoverflow.com/a/6027615/2476373
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _value2str(val):
    if isinstance(val, float):  # and not 1e-3<val<1e3:
        # %g means: "Floating point format.
        # Uses lowercase exponential format if exponent is less than -4 or not less than precision,
        # decimal format otherwise."
        val = '%g' % val
    else:
        val = '{}'.format(val)
    val = re.sub('\.', '_', val)
    return val
