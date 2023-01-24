import sys
sys.path.append("../../src")
import os
import datetime
import pandas as pd
import numpy as np
from example_lorenz import get_lorenz_data
from sindy_utils import library_size
from training import train_network
import tensorflow as tf