"""Tuning budgets are confined to past train and tuning windows."""
from __future__ import annotations
import numpy as np
import optuna
from lightgbm import LGBMClassifier, early_stopping
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

optuna.logging.set_verbosity(optuna.logging.WARNING)


def lightgbm_config(x, y, masks, trials=16):
    train, tune = masks["train"], masks["tune"]
    def objective(trial):
        config = {
            "n_estimators": 500,
            "num_leaves": trial.suggest_categorical("num_leaves", [7, 15, 31]),
            "learning_rate": trial.suggest_float("learning_rate", .015, .08, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 30, 200),
            "reg_lambda": trial.suggest_float("reg_lambda", .1, 30., log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", .01, 10., log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", .65, 1.),
            "subsample": .85, "subsample_freq": 1,
        }
        model = LGBMClassifier(**config, verbosity=-1, n_jobs=4, random_state=2026, metric="average_precision")
        model.fit(x[train], y[train], eval_set=[(x[tune], y[tune])], eval_metric="average_precision",
                  callbacks=[early_stopping(40, first_metric_only=True, verbose=False)])
        trial.set_user_attr("best_iteration", int(model.best_iteration_))
        trial.set_user_attr("config", config)
        return average_precision_score(y[tune], model.predict_proba(x[tune])[:, 1])
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=2026))
    study.optimize(objective, n_trials=trials)
    config = dict(study.best_trial.user_attrs["config"])
    config["n_estimators"] = study.best_trial.user_attrs["best_iteration"]
    return config, study.trials_dataframe()


def fit_lgb(x, y, train, columns, config, seed):
    return LGBMClassifier(**config, verbosity=-1, n_jobs=4, random_state=seed).fit(x[train][:, columns], y[train])


def fit_history(x, y, masks, columns, seed):
    best = None
    for leaves in (15, 31):
        for l2 in (2., 8.):
            model = HistGradientBoostingClassifier(max_iter=180, learning_rate=.04, max_leaf_nodes=leaves,
                        min_samples_leaf=40, l2_regularization=l2, early_stopping=False, random_state=seed)
            model.fit(x[masks["train"]][:, columns], y[masks["train"]])
            ap = average_precision_score(y[masks["tune"]], model.predict_proba(x[masks["tune"]][:, columns])[:,1])
            if best is None or ap > best[0]:
                best = (ap, model)
    return best[1]
