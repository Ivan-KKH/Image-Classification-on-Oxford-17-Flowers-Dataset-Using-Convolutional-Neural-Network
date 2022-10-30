from packaging import version

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import tensorboard as tb

experiment_id = "Oct29_22-33-51_HK-IDC-172-31-100-11ResNet18_SGD_momentum_0.9_lr_0.001_gamma_0.1_50 epochs"
experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
df = experiment.get_scalars()
df