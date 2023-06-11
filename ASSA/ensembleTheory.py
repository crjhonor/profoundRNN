
"""
Illustrate the ensemble theory of the ensemble approach and the majority voting to make myself more understanding of
what I should do.
Key Points:
The error rate of the ensemble is much lower than the error rate of each individual classifier.
"""

from scipy.special import comb
import math
def ensemble_error(n_classifier, error):
    k_state = int(math.ceil(n_classifier / 2.))
    probs = [comb(n_classifier, k) * error ** k * \
             (1-error)**(n_classifier - k) for k in range(k_state, n_classifier+1)]
    return sum(probs)
for i in range(20):
    ens_error = ensemble_error(n_classifier=i+1, error=0.4149)
    print(f"{i+1} classifiers will achieve {1-ens_error} accuracy.")

import numpy as np
import matplotlib.pyplot as plt
error_range = np.arange(0.0, 1.01, 0.01)
ens_errors = [ensemble_error(n_classifier=9, error=error) for error in error_range]
plt.plot(error_range, ens_errors, label='Ensemble error', linewidth=2)
plt.plot(error_range, error_range, linestyle='--', label='Base error', linewidth=2)
plt.xlabel('Base error')
plt.ylabel('Base/Ensemble error')
plt.legend(loc='upper left')
plt.grid(alpha=0.5)
plt.show()
