import mlflow
import mlflow.keras
from tensorflow import keras

model = keras.models.load_model("mlruns/455164518356477147/b5a46c0038e442b5a8d36a7216c0829a/artifacts/best_cnn_model_overall.keras")



# Start an MLflow run
with mlflow.start_run() as run:
    # Log the model to MLflow under the artifact path 'model'
    mlflow.keras.log_model(model, "model")
    
    # Print run ID for reference
    run_id = run.info.run_id
    print("Run ID:", run_id)

    # Register the model under a name
    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri, "Best_Overall_Model")

    print(f"Model registered as 'Best_Overall_Model' (version: {result.version})")
