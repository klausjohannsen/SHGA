# libs
import os
import shutil
from hyperqueue import LocalCluster
from hyperqueue.cluster import WorkerConfig
from hyperqueue import Job
import re

# config
hq_tmp_dir = '.hq_tmp'
rm_hq_tmp_dir = False

# functions
def all_int(s):
    r = [int(ss) for ss in s.split() if ss.isdigit()]
    if len(r) == 1:
        r = r[0]
    return(r)

def all_float(s):
    f1 = '[-+]?[\d]+\.?[\d]*[Ee](?:[-+]?[\d]+)?'
    f2 = '[-+]?[\d]+\.?[\d]*'
    f3 = '[-+]?\.\d+'
    f4 = '[-+]?\d+'
    f = f'{f1}|{f2}|{f3}|{f4}'
    r = re.findall(f, s)
    r = [float(rr) for rr in r]
    return(r)

def last_4_float(s):
    r = all_float(s)
    return(r[-4:])

def last_5_float(s):
    r = all_float(s)
    return(r[-5:])

def last_int(s):
    r = all_int(s)
    return(r[-1])

def hq_run_n(cmd = None, n = 0, pp = None):
    assert(cmd is not None)
    assert(n > 0)

    # create empty .hq_tmp
    shutil.rmtree(hq_tmp_dir, ignore_errors=True)
    os.mkdir(hq_tmp_dir)

    # hq 
    with LocalCluster() as cluster:
        cluster.start_worker()
        client = cluster.client()
        job = Job()
        for k in range(n):
            job_name = f'{hq_tmp_dir}/job_{k}'
            with open(job_name , mode='w') as f:
                f.write(f'{cmd} > {hq_tmp_dir}/out_{k}')
            job.program(["/usr/bin/sh", job_name], stdout = f'{hq_tmp_dir}/stdout_{k}', stderr = f'{hq_tmp_dir}/stderr_{k}')
        submitted = client.submit(job)
        client.wait_for_jobs([submitted])

    # collecting output
    r = []
    for k in range(n):
        fn = f'{hq_tmp_dir}/out_{k}'
        while(True):
            if os.path.getsize(fn) > 0:
                break
        with open(f'{hq_tmp_dir}/out_{k}', mode='r') as f:
            s = f.read()
            r += [" ".join(s.split())]

    # rm .hq_tmp
    if rm_hq_tmp_dir:
        shutil.rmtree(hq_tmp_dir, ignore_errors=True)

    # post-process
    if pp is None: pp = lambda x: x
    if isinstance(pp, str) and pp == 'int': pp = all_int
    if isinstance(pp, str) and pp == 'last_int': pp = last_int
    if isinstance(pp, str) and pp == 'float': pp = lambda x: float(x)
    if isinstance(pp, str) and pp == 'all_float': pp = all_float
    if isinstance(pp, str) and pp == 'last_4_float': pp = last_4_float
    if isinstance(pp, str) and pp == 'last_5_float': pp = last_5_float
    r = [pp(s) for s in r]
    return(r)




