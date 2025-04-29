# Cross validation results

- Metric: figure of merit (TPR when FPR = $0.5\%$)

- Grid search orders
    - Round 1: `max_depth` & `min_child_weight`
    - Round 2: `gamma`
    - Round 3: `subsample` & `colsample_bytree`

- Default
    ```json
    {
        "n_estimators_white": 850,
        "n_estimators_grz": 1300,
        "n_estimators_hybrid": 800,
        "eta": 0.02,
        "max_depth": 10,
        "min_child_weight": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "gamma": 0.1,
    }
    ```