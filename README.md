
# Vision Text Transformer

Vision Text Transformer is a deep learning project that encompasses two powerful models for visual language tasks: image captioning and image question answering. It leverages state-of-the-art architectures to provide accurate and efficient solutions for generating textual descriptions of images and answering questions based on visual content.

## Project Overview

This project consists of two main submodels:

1. **Image Captioning using Pretrained Model**: A streamlined model dedicated to generating captions for images using a ViTEncoder (Vision Transformer Encoder) and GPT2Decoder. This model is designed for ease of training and faster performance, making it ideal for scenarios where only image captioning is required.

2. **Vision Transformer**: A versatile model that extends the capabilities of the image captioning model to include image question answering. This model uses PaliGemma for handling both tasks, providing a more comprehensive solution for vision-language applications.

## Getting Started

### Download Database

For both submodels, download the dataset from the provided link and save it in the root directory of the project.

### Download Model (Vision Transformer)

For the Vision Transformer submodel, download the pre-trained model from the provided link and place it in the root directory.

### Installation

1. **Install Required Libraries**:  
   Install the necessary Python libraries by running:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Models

1. **Image Captioning using Pretrained Model**:
   - This model is easier to train and requires less time.
   - To run the image captioning model, execute the following command:
     ```bash
     python image_captioning_using_pretrained_model.py
     ```

2. **Vision Transformer**:
   - This model can handle both image captioning and image question answering.
   - To run the Vision Transformer model, use the appropriate script based on your operating system:
     - On Windows:
       ```bash
       run.bat
       ```
     - On Linux/MacOS:
       ```bash
       bash run.sh
       ```

### Custom Dataset

You can use your own dataset for training or inference. However, ensure that the folder structure and format match those of the provided dataset for seamless integration.

## Images

(Images will be added here to illustrate the models and their outputs.)
