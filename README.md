# Grapevine-Leaf-Image-Denoising-Autoencoder-CNN-Classification

## Project Overview
This project involves developing an autoencoder model to perform image denoising on a dataset of grapevine leaf images. The primary goal is to create a model that can reduce noise from images while preserving important features, making it useful for image preprocessing in tasks like classification or segmentation.

## Goals
The primary goal of this project is to classify grapevine leaves using a relevant dataset. To achieve this, we have employed various deep learning models, including data augmentation techniques, to enhance the accuracy and robustness of the classification process. 

- **MobileNetV2 Model:** Among the models tested, MobileNetV2 achieved a classification precision of 90%, making it the most effective model in our experiments. This aligns with the state-of-the-art approaches for this dataset, where MobileNetV2 is often used as the backbone to achieve optimal results.

- **Visualization and Statistics:** Relevant images, statistics, and performance metrics are included in the accompanying notebook to provide a clear understanding of the model's effectiveness and the classification process.

## Dataset
The dataset used in this project is the **Grapevine Leaves Image Dataset**. It contains images of grapevine leaves, categorized into different classes. The dataset is stored in the directory structure as follows:

```
Grapevine_Leaves_Image_Dataset/
    ├── Ak/
    ├── Ala_Idris/
    ├── Buzgulu/
    ├── Dimnit/
    └── Nazli/
```

- **Class Labels:** Each subfolder in the dataset directory corresponds to a class of grapevine leaves.

## Dependencies
To run this project, you need the following Python libraries:
- `tensorflow`
- `numpy`
- `pandas`
- `os`
- `seaborn`
- `matplotlib`
- `scikit-learn`
- `opencv-python`

Install these libraries using pip:

```bash
pip install tensorflow numpy pandas seaborn matplotlib scikit-learn opencv-python
```

## Model Architecture
The autoencoder model consists of an encoder and a decoder:
- **Encoder:** The encoder compresses the input images into a latent space of lower dimensionality.
- **Decoder:** The decoder reconstructs the images from the latent space representation.

The model is built using the following layers:
- Convolutional layers with ReLU activation for feature extraction.
- MaxPooling layers for downsampling.
- Flatten and Dense layers for the latent space representation.
- Conv2DTranspose layers for upsampling in the decoder.

## Training
The model is trained to minimize the binary cross-entropy loss between the input images and their reconstructions. The training process involves:
- Splitting the dataset into training and testing sets using an 80/20 split.
- Training the autoencoder for 40 epochs with a batch size of 32.
- Monitoring validation loss to avoid overfitting.

## Results
The training process outputs the loss and validation loss for each epoch, allowing for performance monitoring and model tuning.

## Usage
To use the trained autoencoder model, load your dataset and run the `autoencoder.fit()` function with your data. The model will output the denoised images, which can then be used for further processing.

## Visualization
Sample images from the dataset and their denoised counterparts can be visualized using `matplotlib`. This is useful for qualitatively assessing the performance of the autoencoder.
