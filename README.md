# Predict_animals
link dataset: https://drive.google.com/drive/folders/15wG2QgWU8dKs-NeoLI48TxzFWdydg7Jj

# Predict Animal

**Predict Animal** is an image classification application that uses a deep learning model (ResNet50) to classify animal species based on input images. The goal of this project is to accurately identify animal species from a given dataset of different animal species.

## Description

This project uses PyTorch to build and train an image classification model. The ResNet50 model is used due to its excellent performance in image recognition, especially when pre-trained on the ImageNet dataset. After training, the model can classify animal species from the prepared dataset.

## Features
- **Animal Image Classification**: Uses the ResNet50 model to classify images into different animal species.
- **Predict New Images**: Users can provide an animal image, and the model will return the identified species.
- **Model Training and Evaluation**: The model is trained using the training data and evaluated using a testing dataset.
- **Training Monitoring**: Uses TensorBoard to track training metrics such as loss and accuracy.

## Folder Structure

Predict_Animal/ ├── animals/ │ ├── train/ │ │ ├── butterfly/ │ │ ├── cat/ │ │ ├── chicken/ │ │ ├── cow/ │ │ ├── dog/ │ │ ├── elephant/ │ │ ├── horse/ │ │ ├── sheep/ │ │ ├── spider/ │ │ ├── squirrel/ │ ├── valid/ │ ├── test/ ├── checkpoint/ │ ├── best_cnn.pt │ ├── last_cnn.pt ├── image/ │ ├── your_image.jpg ├── model.py ├── process_image.py ├── predict.py ├── train.py ├── requirements.txt └── README.md

## Parameters

--root: Path to the root directory containing the dataset.

--epoch: Number of epochs for training (default is 10).

--batch: Batch size (default is 128).

--image: Path to the image for prediction.

--checkpoint: Path to the checkpoint model for prediction.

## Train model

python train.py --root <path_to_dataset> --epoch 10 --batch 128

## Predict on a New Image
python predict.py --image <path_to_image> --checkpoint checkpoint/best_cnn.pt

## License

This README now includes clear headings and sections with more structure, making it easy for others to follow. Let me know if you'd like to modify or add anything!

