import numpy as np
import time
from sktime.transformations.panel.rocket import MiniRocketMultivariate
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler

print("Generating random data...")
X = np.random.randn(2000, 20, 656)
y = np.random.randint(0, 10, 2000)

print("Testing MiniRocket transform...")
t0 = time.time()
transform = MiniRocketMultivariate(num_kernels=10000)
X_transformed = transform.fit_transform(X)
print(f"Transform took {time.time()-t0:.2f}s")

print("Testing Scaler...")
t0 = time.time()
scaler = StandardScaler()
X_transformed = scaler.fit_transform(X_transformed)
print(f"Scaler took {time.time()-t0:.2f}s")

print("Testing Ridge...")
t0 = time.time()
classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), class_weight='balanced')
classifier.fit(X_transformed, y)
print(f"Ridge took {time.time()-t0:.2f}s")
