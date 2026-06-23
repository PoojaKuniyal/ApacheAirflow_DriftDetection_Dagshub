RANDOM_FOREST_PARAMETERS = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [None, 10, 20, 30, 40],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None],
    "bootstrap": [True, False],
}

RANDOM_SEARCH_PARAM = {
    "n_iter": 20,
    "cv": 3,
    "scoring": "f1",
    "random_state": 42,
    "n_jobs": -1,
}
