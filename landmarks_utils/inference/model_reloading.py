
import mlflow
from mlflow.tracking import MlflowClient
import os
import torch 



def get_device(requested_device: str | None = None) -> torch.device:
    if requested_device is not None:
        requested_device = requested_device.lower()

    if requested_device in {None, "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() is False. "
            "Use device=cpu or install a PyTorch build compatible with the NVIDIA driver."
        )

    return torch.device(requested_device)


def load_models_and_settings(config):
    if config["model"].get("reload_from_state_dict", False):
        raise NotADirectoryError()
    else: 
        # define mlflow runs directory
        mlflow_runs_dir = config["model"]["mlflow_runs_dir"] # "/home/cwatzenboeck/data/mlflow_cirpc_tmp/RA/data/" 
        os.environ["MLFLOW_TRACKING_URI"] = mlflow_runs_dir
        run_id = config["model"]["run_id"]
        model_artifact_name = config["model"].get("model_artifact_name", "best_model")
        logged_model_uri = f"runs:/{run_id}/{model_artifact_name}"  # or the path you used
        model = mlflow.pytorch.load_model(logged_model_uri)
        model.eval()

        device = get_device(getattr(config, "device", "auto"))
        heatmap_generator_artifact_name = config["model"].get("heatmap_generator_artifact_name", "best_heatmap_generator")
        logged_heatmap_uri = f"runs:/{run_id}/{heatmap_generator_artifact_name}"
        heatmap_generator = mlflow.pytorch.load_model(logged_heatmap_uri, map_location=device)
        heatmap_generator.eval();


        client = MlflowClient()
        run = client.get_run(run_id)
        landmark_names_param = run.data.params.get("landmark_names")
        if landmark_names_param is not None:
            landmark_names = [name.strip().strip("'").strip('"') for name in landmark_names_param.strip("[]").split(",")]
        else:
            landmark_names = None
        N_landmarks = len(landmark_names) if landmark_names != None else None 
        if N_landmarks == None: 
            raise ValueError("Number of landmarks could not be determined. Please verify that 'landmark_names' is provided in the MLflow run parameters or configuration settings. Ensure the parameter is set correctly.")
         
        dim_image_param = run.data.params.get("dim_image")
        if isinstance(dim_image_param, str):
            dim_image_param = dim_image_param.strip("[]")
            dim_image = [int(x.strip()) for x in dim_image_param.split(',') if x.strip()]
        else:
            dim_image = list(map(int, dim_image_param))
         
        settings = {
            #"dim_image": config["input"]["dim_image"],
            # "N_landmarks": config["input"]["N_landmarks"],
            "dim_image": dim_image,
            "N_landmarks": N_landmarks,
            "landmark_names": landmark_names
        } 
        return model, heatmap_generator, settings
    