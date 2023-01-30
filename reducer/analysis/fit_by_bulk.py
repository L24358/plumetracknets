import os

sines = [3, 4, 5, 10, 13, 15, 18, 30, 35, 40, 42, 43, 45, 47, 48, 50, 53, 55, 58, 60, 63, 65]
constants = [68, 82]
 
for i in constants:
    os.system(f"python3 fit_obs_act_manual.py {i}")