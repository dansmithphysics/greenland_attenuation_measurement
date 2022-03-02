import numpy as np
import matplotlib.pyplot as plt

nthrows = 10000
n1 = np.random.normal(1.4, 0.1, nthrows)
n2 = np.random.normal(1.78, 0.03, nthrows)

entries = np.sort(np.square(n2 / n1))
cumsum = np.cumsum(np.ones(len(entries)))
cumsum = np.array(cumsum) / float(cumsum[-1])

entries_min = entries[np.argmin(np.abs(cumsum - (0.5 - 0.341)))]
entries_mid = entries[np.argmin(np.abs(cumsum - (0.5 - 0.000)))]
entries_max = entries[np.argmin(np.abs(cumsum - (0.5 + 0.341)))]

print("%f (+%f)(-%f)" % (entries_mid, entries_mid - entries_min, entries_max - entries_mid))
