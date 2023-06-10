# pytorch_transfer_learning

This repository contains a PyTorch-based image classifier using transfer learning with a pretrained efficientnet_b0 model. It specializes in identifying food images.

## Getting Started
To get started with this project, follow these steps:

1. Clone the repository to your local machine.
2. Install the necessary dependencies.
3. Train the model using the `train.py` script.
4. Make predictions using the trained model with the `predict.py` script.

## Installation
To install the required dependencies, run the following command:
```bash
pip install -r requirements.txt
```

## Training
To train the model, execute the `train.py` script with the desired number of epochs. The default number of epochs is 5. Use the following command:
```bash
python train.py --num_epochs <num_epochs>
```
Replace `<num_epochs>` with the desired number of epochs.

## Making Predictions
To make predictions using the trained model, execute the `predict.py` script with the path to the image you want to classify. Use the following command:
```bash
python predict.py --image_path <image_path>
```
Replace `<image_path>` with the path to the image file.

## Additional Notes
- The pretrained efficientnet_b0 model is used as the base model for transfer learning.
- The `train.py` script performs the training of the model using the provided dataset.
- The `predict.py` script uses the trained model to classify a single image.
- Ensure that you have the necessary dataset available before running the training script.

Feel free to explore and modify the code to fit your specific needs!