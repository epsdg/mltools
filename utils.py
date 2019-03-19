import os
import sys
import gc
import numpy as np
import pandas as pd
from datetime import datetime

pd.options.display.max_columns = 1000
pd.options.display.max_rows = 1000
pd.options.mode.use_inf_as_na = True
pd.options.display.float_format = '{:.3f}'.format
float_formatter = lambda x: "%.4f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

from .config import cfg


def get_logger(name):
    from logging import config, getLogger, FileHandler, Formatter
    config.dictConfig(cfg['logger_config'])
    logger = getLogger()
    t = datetime.now().strftime('%m%d-%H%M')
    fh = FileHandler('mltools_log_' + name + '_' + t + '.txt', mode='w')
    formatter = Formatter('%(asctime)s: %(message)s', datefmt='%H:%M:%S')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def load_hparams(file_in):
    import yaml
    with open(file_in) as f:
        hparams = f.read()
    return yaml.load(hparams, Loader=yaml.Loader)


def cpu_count():
    from multiprocessing import cpu_count
    return cpu_count()


def map_fn(func, args, n_processes=None, backend=None, **kwargs):
    '''Parallel map function.  Allows selection of the backend multiprocessing
    engine to accomodate platform and resources available.

    Args:
        func: callable to execute
        args: iterable with args to pass to func
        n_processes: number of jobs (threads or processes, depending on the value
            of backend).  If None, uses cfg.n_cores_default from .config.py
        backend: string
            'mp': multiprocessing.Pool
            'tpe': ThreadPoolExecutor
            'ppe': ProcessPoolExecutor
            'joblib-threads': joblib, prefer='threads'
            'joblib'-processes: joblib, prefer='processes' (loky backend)
            'dask': dask.delayed
            'dask-dist': dask.distributed.Client

    Returns: list of outputs
    '''

    def mp_map(func, args, n_workers=None):
        from multiprocessing import Pool
        with Pool(n_workers) as p:
            output = p.map(func, args)
        return output

    def tpe_map(func, args, n_processes=None):
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=n_processes) as e:
            output = list(e.map(func, args))
        return output

    def ppe_map(func, args, n_processes=None):
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=n_processes) as e:
            output = e.map(func, args)
        return output

    def joblib_map(func, args, n_jobs=None, prefer='threads', verbose=10):
        from joblib import Parallel, delayed
        output = Parallel(n_jobs=n_jobs, prefer=prefer, verbose=verbose)(
        (delayed(func)(arg) for arg in args))
        return output

    def dask_map(func, args, **kwargs):
        import dask
        job_list = []
        for job in args:
            dl = dask.delayed(func)(job)
            job_list.append(dl)
        output = list(dask.compute(*job_list))
        return output

    def dask_dist_map(func, args, threads_per_worker=1, n_workers=None, **kwargs):
        from dask.distributed import Client, progress
        client = Client(threads_per_worker=threads_per_worker,
                        n_workers=n_workers,
                        **kwargs)
        print('scheduler at', client.scheduler_info()['address'], end='')
        console_ip = client.scheduler_info()['address'].split(':')[1]
        print(', console ', console_ip + ':8787', sep='')
        futures = client.map(func, args)
        output = client.gather(futures)
        client.close()
        return output

    if n_processes is None: n_processes = cfg['n_cores_default']
    if backend is None: backend = cfg['default_map_fn_backend']

    if backend=='mp':
        return mp_map(func, args, n_workers=n_processes, **kwargs)
    elif backend=='tpe':
        return tpe_map(func, args, n_processes=n_processes, **kwargs)
    elif backend=='ppe':
        return ppe_map(func, args, n_processes=n_processes, **kwargs)
    elif backend=='joblib-threads':
        return joblib_map(func, args, n_jobs=n_processes, prefer='threads', **kwargs)
    elif backend=='joblib-processes':
        return joblib_map(func, args, n_jobs=n_processes, prefer='processes', **kwargs)
    elif backend=='dask':
        return dask_map(func, args)
    elif backend=='dask-dist':
        return dask_dist_map(func, args, n_workers=n_processes, **kwargs)
