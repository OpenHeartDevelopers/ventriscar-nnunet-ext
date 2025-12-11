# -*- coding: utf-8 -*-
"""
Prediction script for Myocardial Scar Segmentation using a custom nnU-Net model.

This script is designed to process a single LGE-MRI NIfTI file.
"""

# Apply patch
from ventriscar_nnunet_ext.patching.patcher import apply_nnunet_patch
apply_nnunet_patch() 

import argparse
import time
import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
import SimpleITK as sitk
import torch
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def initialize_predictor(model_dir: Path, use_gpu: bool = False) -> nnUNetPredictor:
    """
    Initializes the nnUNetPredictor with the specified model weights.

    This is a heavy operation and should only be done once per session if
    processing multiple images.

    Args:
        model_dir (Path): The directory containing the trained nnU-Net model files
                          (including plans.json and checkpoint_final.pth).
        use_gpu (bool): If True, attempts to use a CUDA device. Defaults to True.

    Returns:
        nnUNetPredictor: An initialized predictor instance ready for inference.
    """
    logging.info("Initializing model from: %s", model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )

    predictor.initialize_from_trained_model_folder(
        str(model_dir),
        use_folds=('all',),
        checkpoint_name='checkpoint_final.pth',
    )
    logging.info("Model initialized successfully.")
    return predictor


def run_scar_segmentation(
    predictor: nnUNetPredictor, input_path: Path, output_path: Path
) -> None:
    """
    Runs scar segmentation on a single NIfTI file.

    Args:
        predictor (nnUNetPredictor): The initialized nnU-Net predictor instance.
        input_path (Path): Path to the input NIfTI image.
        output_path (Path): Path where the output segmentation will be saved.
    """
    try:
        logging.info("Processing file: %s", input_path)
        # 1. Read image and properties
        img_sitk, props = SimpleITKIO().read_images([str(input_path)])

        # 2. Run prediction
        pred_array = predictor.predict_single_npy_array(
            img_sitk, props, None, None, False
        )
        pred_array = pred_array.astype(np.uint8)
        logging.info("Prediction generated. Unique labels: %s", np.unique(pred_array))

        # 3. Convert back to SimpleITK image and save
        pred_image = sitk.GetImageFromArray(pred_array)
        pred_image.SetDirection(props['sitk_stuff']['direction'])
        pred_image.SetOrigin(props['sitk_stuff']['origin'])
        pred_image.SetSpacing(props['sitk_stuff']['spacing'])

        output_path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(pred_image, str(output_path), useCompression=True)
        logging.info("Segmentation saved to: %s", output_path)

    except Exception as e:
        logging.error("Error processing %s: %s", input_path, e, exc_info=True)


def main(args: argparse.Namespace) -> None:
    """
    Main function to orchestrate the prediction process.
    """
    start_time = time.time()
    input_file = Path(args.input_file)
    output_file = Path(args.output_file) if args.output_file else \
                  input_file.parent / f"seg_{input_file.name}"

    # For now, we use a hardcoded path. Later, this will be managed by pycemrg.
    model_folder_path = Path.home() / '.cache' / 'ventri_scar' / 'segment_lge' / \
                        'nnUNetTrainerUxLSTMEnc__nnUNetPlans__3d_fullres'

    # --- Core Logic ---
    predictor = initialize_predictor(model_folder_path, use_gpu=False)
    run_scar_segmentation(predictor, input_file, output_file)
    # ------------------

    elapsed_time = time.time() - start_time
    logging.info("Total processing time: %.2f seconds.", elapsed_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run myocardial scar segmentation on a NIfTI image."
    )
    parser.add_argument(
        '--input-file',
        type=str,
        required=True,
        help='Path to the input NIfTI file.'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default=None,
        help='Path to save the output segmentation. If not provided, it will be '
             'saved as "seg_<input_filename>" in the same directory as the input.'
    )
    parsed_args = parser.parse_args()
    main(parsed_args)
