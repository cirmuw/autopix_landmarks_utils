import numpy as np
import pandas as pd
import torch
import pydicom
import matplotlib.pyplot as plt
import argparse
from typing import Union, List, Tuple
from pathlib import Path

import landmarks_utils.data.data_utils







# def plot_landmarks(image: str | Path | torch.Tensor, 
#                    landmarks, 
#                    landmarks_targets=None, 
#                    figsize=(12, 10)):
#     """
#     Plots an image (either from a file path, a NumPy array, or a PyTorch tensor) with landmarks.

#     Args:
#         image (Union[str, np.ndarray, torch.Tensor]): Image source, either a DICOM file path, 
#                                                       a NumPy array, or a PyTorch tensor.
#         landmarks (np.ndarray): Numpy array of shape (N, 2), where N is the number of landmarks.
#                                 Each row represents (x, y) coordinates.
#         landmarks_targets (np.ndarray, optional): Ground truth landmarks of the same shape (N, 2).
#                                                   If provided, they will be plotted in blue.

#     Returns:
#         matplotlib.figure.Figure: The figure object containing the plot.
#     """

#     # Load the image if a path is provided
#     if isinstance(image, (str, Path)):  
#         dcm = pydicom.dcmread(image)
#         image = dcm.pixel_array

#     # Convert tensor to numpy array if necessary
#     if isinstance(image, torch.Tensor):
#         image = image.squeeze().cpu().numpy()

#     # Ensure image is in 2D format
#     assert len(image.shape) == 2, "Image should be a 2D grayscale array"

#     # Ensure landmarks are properly shaped
#     assert landmarks.shape[1] == 2, "Landmarks should have shape (N, 2)"

#     if landmarks_targets is not None:
#         assert landmarks_targets.shape == landmarks.shape, \
#             "landmarks_targets must have the same shape as landmarks"

#     # Create figure and axis
#     fig, ax = plt.subplots(figsize=figsize)
#     ax.imshow(image, cmap='gray')

#     # Plot detected landmarks
#     ax.scatter(landmarks[:, 0], landmarks[:, 1], s=20, color='red', marker='x', label="Predicted")

#     # Plot target landmarks if provided
#     if landmarks_targets is not None:
#         ax.scatter(landmarks_targets[:, 0], landmarks_targets[:, 1], 
#                    s=20, color='blue', marker='o', label="Target")

#     # Hide axes and add legend if needed
#     ax.axis("off")
#     if landmarks_targets is not None:
#         ax.legend()

#     return fig  # Return the figure object



def plot_landmarks(
    image: Union[str, Path, torch.Tensor, np.ndarray],
    landmarks: np.ndarray,
    landmarks_targets: np.ndarray | None = None,
    *,
    figsize: Tuple[int, int] = (12, 10),
    annotate: bool = False,
    landmark_labels: List[str] | None = None,
    label_offset: Tuple[int, int] = (4, 4),
    label_fontsize: int = 10,
    label_color: str = "yellow",
):
    """Plot an image with predicted (and optionally target) landmarks, with optional labels.

    Args:
        image (Union[str, Path, torch.Tensor, np.ndarray]): Image source. Can be a file path,
            a NumPy array, or a PyTorch tensor.
        landmarks (np.ndarray): Array of shape (N, 2) with (x, y) coordinates of predicted landmarks.
        landmarks_targets (np.ndarray, optional): Optional ground‑truth landmarks of shape (N, 2).
        figsize (Tuple[int, int], optional): Figure size in inches.
        annotate (bool, optional): If *True*, annotate each landmark with its index (default) or a
            corresponding value from *landmark_labels*.
        landmark_labels (List[str], optional): List of custom labels for each landmark. Must be the
            same length as *landmarks* if provided. Ignored when *annotate* is *False*.
        label_offset (Tuple[int, int], optional): Pixel offset (dx, dy) for the text to avoid overlap.
        label_fontsize (int, optional): Font size of the labels.
        label_color (str, optional): Color of the label text.

    Returns:
        matplotlib.figure.Figure: The resulting figure.
    """

    # --- Load / normalize image -------------------------------------------------
    if isinstance(image, (str, Path)):
        dcm = pydicom.dcmread(image)
        image = dcm.pixel_array

    if isinstance(image, torch.Tensor):
        image = image.squeeze().cpu().numpy()

    assert image.ndim == 2, "Image should be a 2‑D grayscale array"
    assert landmarks.shape[1] == 2, "Landmarks should have shape (N, 2)"

    if landmarks_targets is not None:
        assert landmarks_targets.shape == landmarks.shape, (
            "landmarks_targets must have the same shape as landmarks"
        )

    if annotate and landmark_labels is not None:
        assert len(landmark_labels) == len(landmarks), (
            "landmark_labels must match the number of landmarks"
        )

    # --- Plot -------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image, cmap="gray")
    ax.axis("off")

    # Predicted landmarks
    ax.scatter(landmarks[:, 0], landmarks[:, 1], s=20, color="red", marker="x", label="Predicted")

    # Target landmarks
    if landmarks_targets is not None:
        ax.scatter(
            landmarks_targets[:, 0],
            landmarks_targets[:, 1],
            s=20,
            color="blue",
            marker="o",
            label="Target",
        )

    # Optional legend (only when ground truth is present)
    if landmarks_targets is not None:
        ax.legend()

    # Optional annotation --------------------------------------------------------
    if annotate:
        for idx, (x, y) in enumerate(landmarks):
            text = landmark_labels[idx] if landmark_labels is not None else str(idx)
            ax.text(
                x + label_offset[0],
                y + label_offset[1],
                text,
                fontsize=label_fontsize,
                color=label_color,
                va="bottom",
                ha="left",
                bbox=dict(boxstyle="round,pad=0.15", fc="black", ec="none", alpha=0.0),
            )

    return fig



def plot_landmarks_from_df(dfm, image_idx=0):
    """
    Extracts the image path and landmarks from the DataFrame and plots them.

    Args:
        dfm (pd.DataFrame): DataFrame containing image paths and landmarks.
        image_idx (int): Index of the image to be plotted.

    Returns:
        matplotlib.figure.Figure: The figure object containing the plot.
    """
    # Extract image path and landmarks
    image_path = dfm["image"].iloc[image_idx]
    landmarks = landmarks_utils.data.data_utils.extract_landmarks_from_df(dfm, image_idx)

    # Plot the image with landmarks
    return plot_landmarks(image_path, landmarks)


def _infer_landmark_names(dfm: pd.DataFrame) -> List[str] | None:
    """Infer landmark base names from DataFrame columns."""
    not_lm_columns = ["image_path", "img"]
    lm_columns = [c for c in dfm.columns if c not in not_lm_columns]
    
    
    x_cols = [col for col in lm_columns if col.endswith("-X")]
    y_cols = [col for col in lm_columns if col.endswith("-Y")]
    missed = (set(lm_columns) - set(x_cols)) - set(y_cols)
    if len(missed) != 0: 
        print(" WARNING::: There are missed unexpected columns in the csv file!!")
    
    if x_cols:
        return [col[:-2] for col in x_cols]

    xy_cols = [col[:-2] for col in x_cols]
    if xy_cols:
        return xy_cols

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot landmarks from a CSV file.")
    #parser.add_argument("--landmarks_csv", type=str, required=True, help="Path to landmarks CSV file.")
    # Debugging:
    parser.add_argument("--landmarks_csv", 
                        type=str, required=False, 
                        default="/home/cwatzenboeck/code/RA/autopix_muw_x_ray_pipeline/dev_dir/output_dir/AUTOPIX_000017_20170505_F_L_dp_MTwo_landmarks_F.csv",
                        help="Path to landmarks CSV file.")
    
    parser.add_argument("--idx", type=int, default=0, help="Row index in CSV to plot.")

    
    

    args = parser.parse_args()

    df_landmarks = pd.read_csv(args.landmarks_csv)
    landmark_names = _infer_landmark_names(df_landmarks)

    if landmark_names:
        landmarks = landmarks_utils.data.data_utils.extract_landmarks_from_df(
            df_landmarks,
            image_idx=args.idx,
            landmark_names=landmark_names,
        )
        label_names = landmark_names
    else:
        landmarks = landmarks_utils.data.data_utils.extract_landmarks_from_df(
            df_landmarks,
            image_idx=args.idx,
        )
        label_names = [str(i) for i in range(len(landmarks))]


    image_path = df_landmarks["image_path"].iloc[args.idx]
    plot_landmarks(
        image=image_path,
        landmarks=landmarks,
        annotate=True,
        landmark_labels=label_names,
    )
    plt.show()
