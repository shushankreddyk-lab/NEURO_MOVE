import numpy as np
import time
from sktime.transformations.panel.rocket import MiniRocketMultivariate

X = np.random.randn(2000, 20, 656)
transform = MiniRocketMultivariate(num_kernels=10000)

print("First transform (includes JIT compilation)...")
t0 = time.time()
transform.fit_transform(X)
print(f"First transform took {time.time()-t0:.2f}s")

print("Second transform (compiled)...")
t0 = time.time()
transform.fit_transform(X)
print(f"Second transform took {time.time()-t0:.2f}s")
