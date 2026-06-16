from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from monai.data import DataLoader
from monai.transforms import Compose, ScaleIntensityd, Transposed

import landmarks_utils.data.data_utils
import landmarks_utils.inference.landmarks_inference_utils
from landmarker.data.landmark_dataset import LandmarkDatasetOnTheFly
from landmarks_utils.inference.model_reloading import get_device, load_models_and_settings


def collect_image_paths_from_folder(image_folder: str | Path) -> list[str]:
    folder = Path(image_folder)
    image_paths = sorted(folder.glob("*.dcm"))
    image_paths = [p for p in image_paths if not p.name.startswith("._")]
    return [str(path) for path in image_paths]


def build_inference_transform() -> Compose:
    return Compose(
        [
            Transposed(keys=["image"], indices=(0, 2, 1)),
            ScaleIntensityd(("image",)),
        ]
    )


def _build_dummy_landmarks(num_images: int, num_landmarks: int) -> np.ndarray:
    dummy_landmarks = np.zeros((num_images, num_landmarks, 2), dtype=np.uint16)
    if num_landmarks > 1:
        dummy_landmarks[:, 1, 0] = 1
    if num_landmarks > 2:
        dummy_landmarks[:, 2, 1] = 1
    return dummy_landmarks


def predict_from_image_paths(config, image_paths: Sequence[str]) -> pd.DataFrame:
    if not image_paths:
        raise ValueError("No input image paths were provided.")

    model, _heatmap_generator, settings = load_models_and_settings(config)
    num_landmarks = settings["N_landmarks"]
    dim_image = settings["dim_image"]
    landmark_names = settings["landmark_names"]

    image_paths, image_paths_errors = landmarks_utils.data.data_utils.filter_image_paths(list(image_paths))
    print(f"Filtered image paths: {len(image_paths)}")
    print(f"Filtered image paths with errors: {len(image_paths_errors)}")
    if image_paths_errors:
        print(f"Errors while reading DICOM files: {image_paths_errors}")

    if not image_paths:
        raise ValueError("No valid DICOM files remained after filtering.")

    ds = LandmarkDatasetOnTheFly(
        image_paths,
        _build_dummy_landmarks(num_images=len(image_paths), num_landmarks=num_landmarks),
        pixel_spacing=None,
        transform=build_inference_transform(),
        dim_img=dim_image,
    )
    loader = DataLoader(ds, batch_size=1, num_workers=0, shuffle=False)

    device = get_device(getattr(config, "device", "auto"))
    (
        _all_pred_landmarks,
        all_pred_landmarks_transformed,
        _all_dim_origs,
        _all_pixel_spacings,
        _all_paddings,
    ) = landmarks_utils.inference.landmarks_inference_utils.predict_landmarks(
        model, loader, device=device
    )
    return predictions_to_dataframe(
        image_paths=image_paths,
        all_pred_landmarks_transformed=all_pred_landmarks_transformed,
        landmark_names=landmark_names,
    )


def predictions_to_dataframe(
    image_paths: Sequence[str],
    all_pred_landmarks_transformed: np.ndarray,
    landmark_names: Sequence[str],
) -> pd.DataFrame:
    lm_coordinate_names_generic = [
        f"landmark__{i}_{axis}"
        for i in range(all_pred_landmarks_transformed.shape[1])
        for axis in ["x", "y"]
    ]
    column_names = ["image_path"] + lm_coordinate_names_generic
    landmarks_flat = [list(np.array(landmark).flatten()) for landmark in all_pred_landmarks_transformed]
    landmarks_df = pd.DataFrame(
        [[image_paths[i]] + landmarks_flat[i] for i in range(len(image_paths))],
        columns=column_names,
    )
    landmarks_df["img"] = landmarks_df["image_path"].apply(lambda x: Path(x).stem)

    lm_coordinate_names = [f"{lm}-{axis}" for lm in landmark_names for axis in ["X", "Y"]]
    renaming_dict = {k: v for k, v in zip(lm_coordinate_names_generic, lm_coordinate_names)}
    landmarks_df.rename(columns=renaming_dict, inplace=True)
    return landmarks_df


def default_single_output_path(dcm_file: str | Path, extremity: str) -> Path:
    dcm_path = Path(dcm_file)
    return dcm_path.with_name(f"{dcm_path.stem}_landmarks_{extremity}.csv")


def resolve_folder_output_path(config) -> Path:
    output_dst = config.get("output_dst")
    if not output_dst:
        raise ValueError("`output_dst` must be set for folder inference.")
    output_path = Path(output_dst)
    if str(output_dst).endswith(os.sep) or output_path.suffix == "":
        extremity = config["model"].get("extremity", "unknown")
        output_path = output_path / f"landmarks_{extremity}.csv"
    return output_path


def resolve_single_output_path(config) -> Path:
    dcm_file = config.get("dcm_file")
    if not dcm_file:
        raise ValueError("`dcm_file` must be set for single-image inference.")

    extremity = config["model"].get("extremity", "unknown")
    default_path = default_single_output_path(dcm_file=dcm_file, extremity=extremity)
    output_dst = config.get("output_dst")

    if not output_dst:
        return default_path

    output_path = Path(output_dst)
    if str(output_dst).endswith(os.sep) or output_path.suffix == "":
        return output_path / default_path.name
    return output_path


def save_predictions_csv(
    landmarks_df: pd.DataFrame,
    output_path: str | Path,
    overwrite_output: bool = False,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not overwrite_output:
        raise FileExistsError(
            f"The output file already exists: {output_path}. "
            "Set `overwrite_output=true` to allow replacing it."
        )

    landmarks_df.to_csv(output_path, index=False)
    print(f"Saved landmark predictions to {output_path}")
    return output_path
