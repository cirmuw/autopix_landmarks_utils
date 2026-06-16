from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from landmarks_utils.inference.landmark_prediction import (
    collect_image_paths_from_folder,
    predict_from_image_paths,
    resolve_folder_output_path,
    save_predictions_csv,
)


@hydra.main(version_base=None, config_path="config_landmarks", config_name="F_inference_580_cases_dev")
def main(config: DictConfig) -> None:
    print(OmegaConf.to_yaml(config))

    image_folder = Path(config["input"]["image_folder"])
    image_paths = collect_image_paths_from_folder(image_folder)
    print(f"Files in: {image_folder}   len(image_paths)={len(image_paths)}")

    if config.get("debugging", False):
        print("DEBUGGING = TRUE; running only on first 10 images")
        image_paths = image_paths[:10]

    landmarks_df = predict_from_image_paths(config=config, image_paths=image_paths)
    output_path = resolve_folder_output_path(config=config)
    save_predictions_csv(
        landmarks_df=landmarks_df,
        output_path=output_path,
        overwrite_output=bool(config.get("overwrite_output", False)),
    )


if __name__ == "__main__":
    main()
    print("Done")
