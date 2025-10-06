import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def get_model(X=None, y=None, use_gridsearch=False):
    if use_gridsearch and X is not None and y is not None:
        param_grid = {
        'n_estimators':[100, 200, 300],
        'max_depth':[10],
        'max_features':['sqrt', 'log2'],
        'min_samples_leaf': [1,5]
        }
        
        rf = RandomForestClassifier(
             n_jobs=-1,
             random_state =42
        )

        grid = GridSearchCV (
             estimator=rf,
             param_grid=param_grid,
             cv=3,
             scoring='f1_macro',
             n_jobs=1,
             verbose=2
        )
        grid.fit(X, y)

        print("Best Parameters: ", grid.best_params_)
        return grid.best_estimator_

    #fallback: baseline model
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        n_jobs=-1,
        random_state=42
    )


def save_model(clf, path: str):
    with open(path, 'wb') as f:
        pickle.dump(clf, f)


def load_model(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)
    
