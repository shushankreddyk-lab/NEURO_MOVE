import numpy as np
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import Pipeline
import mne
from mne.decoding import CSP
import pickle

class CSP_Engine:
    def __init__(self, classifier_type="svm", n_components=4):
        self.classifier_type = classifier_type.lower()
        self.n_components = n_components
        self.csp = CSP(n_components=self.n_components, reg='ledoit_wolf', log=True, norm_trace=False)
        
        if self.classifier_type == "svm":
            self.clf = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True)
        else:
            self.clf = LDA(solver='lsqr', shrinkage='auto')
            
        self.pipeline = Pipeline([('CSP', self.csp), (self.classifier_type.upper(), self.clf)])
        
        self.label_classes_ = None

    def fit(self, X, y, X_val=None, y_val=None, progress_callback=None):
        # MNE CSP expects X of shape (n_epochs, n_channels, n_times)
        if X.ndim == 4:
            X = X.squeeze(1) # Remove the singleton channel dim if it's (Batch, 1, Channels, Samples)
            
        n_out = len(np.unique(y))
        y = np.asarray(y)
        
        uniq = np.unique(y)
        if uniq.min() < 0 or uniq.max() >= n_out or len(uniq) != n_out:
            self.label_classes_ = uniq
            lut = {c: i for i, c in enumerate(uniq)}
            y = np.array([lut[v] for v in y], dtype=np.int64)
            n_out = len(uniq)
        else:
            self.label_classes_ = np.arange(n_out)

        self.pipeline.fit(X, y)
        
        if progress_callback:
            # Predict to get training accuracy
            y_pred = self.pipeline.predict(X)
            train_acc = np.mean(y_pred == y)
            
            val_acc = 0.0
            if X_val is not None:
                if X_val.ndim == 4:
                    X_val = X_val.squeeze(1)
                if hasattr(self, 'label_classes_') and self.label_classes_ is not None:
                    lut = {c: i for i, c in enumerate(self.label_classes_)}
                    y_val = np.array([lut.get(v, 0) for v in np.asarray(y_val)], dtype=np.int64)
                y_val_pred = self.pipeline.predict(X_val)
                val_acc = np.mean(y_val_pred == y_val)
                
            progress_callback(1, 0.0, train_acc, 0.0, val_acc)
            
        return self

    def predict(self, X):
        if X.ndim == 4:
            X = X.squeeze(1)
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        if X.ndim == 4:
            X = X.squeeze(1)
        return self.pipeline.predict_proba(X)

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'pipeline': self.pipeline,
                'classifier_type': self.classifier_type,
                'n_components': self.n_components,
                'label_classes': self.label_classes_
            }, f)

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.pipeline = data['pipeline']
            self.classifier_type = data['classifier_type']
            self.n_components = data['n_components']
            self.label_classes_ = data['label_classes']
        return self
