import pickle
from sklearn.ensemble import RandomForestClassifier


def get_model(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42):
    """
    Instantiate a RandomForestClassifier for segmentation.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=n_jobs,
        random_state=random_state
    )


def save_model(clf, path: str):
    with open(path, 'wb') as f:
        pickle.dump(clf, f)


def load_model(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)