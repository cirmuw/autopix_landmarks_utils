from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from landmarks_utils.inference.landmark_prediction import (
    predict_from_image_paths,
    resolve_single_output_path,
    save_predictions_csv,
)


@hydra.main(version_base=None, config_path="config_landmarks", config_name="config")
def main(config: DictConfig) -> None:
    print(OmegaConf.to_yaml(config))

    dcm_file = config.get("dcm_file")
    if not dcm_file:
        raise ValueError("`dcm_file` is required for single-image inference.")

    dcm_path = Path(dcm_file)
    if not dcm_path.exists():
        raise FileNotFoundError(f"Input DICOM file does not exist: {dcm_path}")

    landmarks_df = predict_from_image_paths(config=config, image_paths=[str(dcm_path)])
    output_path = resolve_single_output_path(config=config)
    save_predictions_csv(
        landmarks_df=landmarks_df,
        output_path=output_path,
        overwrite_output=bool(config.get("overwrite_output", False)),
    )


if __name__ == "__main__":
    main()
    print("Done")
