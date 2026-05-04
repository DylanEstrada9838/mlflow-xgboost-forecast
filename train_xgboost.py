from sklearn.metrics import r2_score
import pickle
import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from feature_engineering import get_weekly_df, split_train_test
import optuna
import numpy as np



mlflow.set_experiment("Hyperparameter Tuning XGBoost 2017 30 trials_V2")

def run_training(
    train_start_date, train_end_date, test_start_date, test_end_date
):
    features_df = get_weekly_df()

    train_df, test_df = split_train_test(
        features_df, train_start_date, train_end_date, test_start_date, test_end_date
    )


    X_train = train_df.drop("sales", axis=1)
    y_train = train_df["sales"]

    X_test = test_df.drop("sales", axis=1)
    y_test = test_df["sales"]

    def objective(trial):
        params = {
            "n_estimators":   trial.suggest_int("n_estimators", 50, 300),
            "max_depth":      trial.suggest_int("max_depth", 2, 8),
            "learning_rate":  trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":      trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":      trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":     trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "objective":      "reg:squarederror",
            "eval_metric":    "rmse",
            "random_state":   42
            }
        #Child
        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}") as child_run:
            # Log current trial's parameters
            mlflow.log_params(params)
        
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            r2 = r2_score(y_test, y_pred)

            # Log current trial's error metric
            mlflow.log_metrics({"rmse": rmse, "mape": mape, "r2": r2})

            # Log the model file
            mlflow.xgboost.log_model(model, artifact_path="model")
            
            # Make it easy to retrieve the best-performing child run later
            trial.set_user_attr("run_id", child_run.info.run_id)
            return rmse

    #Parent
    with mlflow.start_run(run_name="optuna_xgboost_regression") as parent_run:
        mlflow.set_tag("model",     "xgboost")
        mlflow.set_tag("optimizer", "optuna")
        mlflow.set_tag("task",      "regression")
        mlflow.set_tag("dataset",   "sales_stores")
        mlflow.set_tag("Scope", "All Stores and Items")

        # Log parameters with MLflow
        mlflow.log_param("train_start_date", train_start_date)
        mlflow.log_param("train_end_date", train_end_date)
        mlflow.log_param("test_start_date", test_start_date)
        mlflow.log_param("test_end_date", test_end_date)
        
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=30)

        # Log the best trial and its run ID
        best = study.best_trial
        mlflow.log_params(best.params)
        mlflow.log_metrics({"best_rmse": best.value})
        if best_run_id := best.user_attrs.get("run_id"):
            mlflow.log_param("best_child_run_id", best_run_id)
        


if __name__ == "__main__":
    run_training(
        train_start_date="2014-01-01",
        train_end_date="2016-12-31",
        test_start_date="2017-01-01",
        test_end_date="2017-12-31",
    )
