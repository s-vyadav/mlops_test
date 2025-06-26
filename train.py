import os
import warnings
import numpy as np
import tensorflow as tf
from PIL import Image
from glob import glob
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from keras_tuner import RandomSearch, HyperModel

import mlflow
import mlflow.keras
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature

# ============== Suppress Warnings and Logs ==============
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", category=UserWarning)

# ============== Safe MLflow Experiment Setup ==============
client = MlflowClient()
base_experiment_name = "TrashNet_Hyperparameter_Tuning"
experiment = client.get_experiment_by_name(base_experiment_name)

if experiment is None:
    mlflow.set_experiment(base_experiment_name)
    experiment_name = base_experiment_name
elif experiment.lifecycle_stage == "deleted":
    # Avoid conflict with deleted experiment
    experiment_name = base_experiment_name + "_v2"
    mlflow.set_experiment(experiment_name)
else:
    experiment_name = base_experiment_name
    mlflow.set_experiment(experiment_name)

print(f"Using MLflow Experiment: {experiment_name}")
print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")

# ============== Data Preparation ==============
classes = {'cardboard': 1, 'glass': 1, 'metal': 1, 'paper': 1, 'plastic': 1, 'trash': 0}
image_paths, labels = [], []

for cls in classes:
    for path in glob(f"data/{cls}/*"):
        image_paths.append(path)
        labels.append(classes[cls])

image_paths = np.array(image_paths)
labels = np.array(labels)

def load_images(paths):
    X = []
    for p in paths:
        img = Image.open(p).resize((64, 64)).convert('RGB')
        X.append(np.array(img) / 255.0)
    return np.array(X)

# ============== CNN HyperModel ==============
class CNNHyperModel(HyperModel):
    def build(self, hp):
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(64, 64, 3)),
            tf.keras.layers.Conv2D(
                filters=hp.Int("conv1_filters", 32, 64, step=32),
                kernel_size=3,
                activation='relu'
            ),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(
                units=hp.Int("dense_units", 64, 128, step=32),
                activation='relu'
            ),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(hp.Choice("learning_rate", [1e-3, 1e-4])),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

# ============== Cross-Validation + Logging ==============
kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
best_models = []
fold_num = 1

for train_idx, val_idx in kf.split(image_paths, labels):
    print(f"\n===== Fold {fold_num} =====")
    X_train = load_images(image_paths[train_idx])
    y_train = labels[train_idx]
    X_val = load_images(image_paths[val_idx])
    y_val = labels[val_idx]

    hypermodel = CNNHyperModel()
    tuner = RandomSearch(
        hypermodel,
        objective='val_accuracy',
        max_trials=3,
        executions_per_trial=1,
        directory='kt_cv_results',
        project_name=f'trashnet_fold_{fold_num}'
    )

    tuner.search(X_train, y_train, epochs=5, validation_data=(X_val, y_val), verbose=1)

    best_model_for_fold = tuner.get_best_models(num_models=1)[0]
    best_trial = tuner.oracle.get_best_trials(1)[0]
    best_score = best_trial.score

    os.makedirs("models", exist_ok=True)
    model_path = f"models/best_model_fold_{fold_num}.keras"
    best_model_for_fold.save(model_path, include_optimizer=False)
    print(f"Saved model to {model_path}")

    with mlflow.start_run(run_name=f"Fold_{fold_num}_Trial{best_trial.trial_id}") as run:
        for param, val in best_trial.hyperparameters.values.items():
            mlflow.log_param(param, val)

        mlflow.log_metric("val_accuracy", best_score)

        summary_path = f"models/model_summary_fold_{fold_num}.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            best_model_for_fold.summary(print_fn=lambda x: f.write(x + "\n"))
        mlflow.log_artifact(summary_path)

        y_pred_probs = best_model_for_fold.predict(X_val).flatten()
        y_preds = (y_pred_probs > 0.5).astype(int)

        acc = accuracy_score(y_val, y_preds)
        prec = precision_score(y_val, y_preds)
        rec = recall_score(y_val, y_preds)
        f1 = f1_score(y_val, y_preds)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        signature = infer_signature(X_val[:1], best_model_for_fold.predict(X_val[:1]))

        mlflow.keras.log_model(
            model=best_model_for_fold,
            artifact_path=f"model_fold_{fold_num}",
            signature=signature
        )
        print(f"Logged model under 'model_fold_{fold_num}'.")

    best_models.append((best_model_for_fold, best_score))
    fold_num += 1

# ============== Save Best Overall Model ==============
best_models.sort(key=lambda x: x[1], reverse=True)
best_overall_model, best_score = best_models[0]
overall_model_path = "models/best_cnn_model_overall.keras"
best_overall_model.save(overall_model_path, include_optimizer=False)

with mlflow.start_run(run_name="Best_Model_Overall") as run:
    mlflow.log_metric("best_overall_val_accuracy", best_score)
    mlflow.log_artifact(overall_model_path)

    y_pred_probs = best_overall_model.predict(X_val).flatten()
    y_preds = (y_pred_probs > 0.5).astype(int)

    acc = accuracy_score(y_val, y_preds)
    prec = precision_score(y_val, y_preds)
    rec = recall_score(y_val, y_preds)
    f1 = f1_score(y_val, y_preds)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    signature = infer_signature(X_val[:1], best_overall_model.predict(X_val[:1]))

    mlflow.keras.log_model(
        model=best_overall_model,
        artifact_path="best_model_overall",
        signature=signature
    )
    print("Best overall model logged.")