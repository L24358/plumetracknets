import os
import reducer.support.navigator as nav
from reducer.config import modelpath

# remove excessive .npy files in modelpath
if 0:
    files = nav.file_finder(target="_type", extension=".npy")
    print(f"Number of files found: {len(files)}")
    for file in files: os.remove(file)

# move files from modelpath to rnn_matrix directory
if 0:
    files = nav.file_finder(target="_type", extension=".npy")
    print(f"Number of files found: {len(files)}")
    for file in files: os.replace(file, os.path.join(modelpath, "rnn_matrix", os.path.basename(file)))