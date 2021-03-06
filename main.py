import platform

import numpy as np
import pandas as pd

from utils import tf_settings

tf = tf_settings(silent=True)
print(f'Python Version: {platform.python_version()}')
print(f'Numpy Version: {np.__version__}')
print(f'Pandas Version: {pd.__version__}')
print(f'TensorFlow Version: {tf.__version__}')
