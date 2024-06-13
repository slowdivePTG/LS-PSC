# Cross validation results

- Metric: figure of merit (TPR when FPR = $0.5\%$)

- Grid search orders
    - Round 1: `max_depth`
    - Round 2: `min_child_weight`
    - Round 3: `subsample` & `colsample_bytree`
    - Round 4: `gamma`

- Default
    ```json
    {
        "n_estimators": 500,
        "eta": 0.05,
        "max_depth": 10,
        "min_child_weight": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "gamma": 0.1,
        "reg_alpha": 0,
        "reg_lambda": 0
    }
    ```