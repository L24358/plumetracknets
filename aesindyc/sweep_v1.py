import os
import time
import concurrent.futures
import reducer.support.basics as bcs
from itertools import product
from aesindyc.config.config_sweep_v1 import argvs

def exec(k):
    argv = pms[k]
    argument = " ".join(["python3", "exec_v1.py"] + bcs.to_string(argv) + ["sweep_v1"])
    os.system(argument)

if __name__=='__main__':
    
    start = time.perf_counter()

    pms = list(product(*argvs.values()))
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(exec, range(len(pms)))

    finish = time.perf_counter()
    print('Total time: '+str(finish-start))