# AVSE Challenge 3

Our entry for the AVSE Challenge 3 consists of a multimodal model that combines visual and textual information to predict the enhanced speech audio. The model is based on a pre-trained Construct Separable 3D CNN model for video extraction and a Deep Complex U-Net for the audio feature extraction. The visual and audio features are fused together in the innermost layer of the UNet, through a cross-attention mechanism before decoding. the model can be training both with si_snr_loss and wsdr_fn losses.

## Requirements

- The data should be placed in the `data` folder.
- Before the execution, the following folders should be created:
    - `output` folder: Contains the generated outputs.
    - `log` folder: Stores the log files.
    - `checkpoints` folder: Saves the model checkpoints.

## Usage

To train the model, run `train.py`.

To generate testing items, execute `test.py`.

To generate evaluation metrics, use `evaluation/objective_evaluation.py`.
