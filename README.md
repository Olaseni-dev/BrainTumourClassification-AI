# Brain Tumour Classification

## Project Description

This project is a comprehensive machine learning example demonstrating how to classify brain tumours using various deep learning models. The project includes steps for setting up a virtual environment, installing necessary packages, and running a machine learning model using Python.

## Setup

To set up the project environment, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Olaseni-dev/BrainTumourClassification-AI.git
   cd BrainTumourClassification
   ```

2. **Create a virtual environment:**
   ```bash
   python3 -m venv myenv
   ```

3. **Activate the virtual environment:**
   - On macOS and Linux:
     ```bash
     source myenv/bin/activate
     ```
   - On Windows:
     ```bash
     myenv\Scripts\activate
     ```

4. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```
## Data Directory

The data directory should be structured as follows:
```
data/
|-- train/
|   |-- 1/
|   |-- 2/
|   ...
|-- test/
|   |-- 1/
|   |-- 2/
|   ...
```
Each subdirectory (`1`, `2`, etc.) contains images of brain tumours classified into respective categories.

## Data Manipulation

Data preprocessing steps include:
- Resizing images to a fixed size suitable for the CNN models.
- Normalizing pixel values to a range of 0 to 1.
- Augmenting the training data with random transformations (e.g., rotations, flips) to improve model generalization.

Example code snippet for data augmentation:
```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
```

## Training and Testing

The dataset is split into training and testing sets. The model is trained on the training set and evaluated on the testing set.

Training steps include:
1. Loading and preprocessing the training data.
2. Defining the CNN model architecture.
3. Compiling the model with an appropriate optimizer and loss function.
4. Training the model using the training data and validating on a validation set.

Testing involves:
1. Loading and preprocessing the test data.
2. Using the trained model to make predictions on the test set.
3. Evaluating the model performance using metrics such as accuracy and confusion matrix.

## CNN Model Used

### Model Architecture
The Convolutional Neural Network (CNN) model used in this project consists of several layers, including:
- Convolutional layers with ReLU activation
- MaxPooling layers
- Flatten layer
- Dense (fully connected) layers

Example code snippet for the CNN model:
```python
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])
```

### VGG-16
VGG-16 is a pre-trained model that has been trained on a large dataset (ImageNet). It is used as a feature extractor in this project to improve the accuracy of the classification model.

Example code snippet for using VGG-16:
```python
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in vgg16.layers:
    layer.trainable = False

model = Sequential([
    vgg16,
    Flatten(),
    Dense(256, activation='relu'),
    Dense(num_classes, activation='softmax')
])
```

## Usage

To run the notebook, follow these steps:

1. **Start Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

2. **Open the `BrainTumourClassification.ipynb` notebook:**
   Navigate to the directory where the notebook is located and open it in your web browser.

3. **Run the notebook:**
   Follow the instructions and run each cell sequentially to train and evaluate the brain tumour classification models.

## Libraries Used

The following libraries are used in this project:
- keras
- tensorflow
- sklearn
- matplotlib
- numpy
- pandas
- plotly
- cv2
- imutils
- itertools
- os
- shutil
- warnings

## Contributing

Contributions are welcome! If you have any improvements or suggestions, please create a pull request or open an issue.

