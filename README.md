# SER517 Group 35 Capstone Project

## Project Scope

The goal of this research project is to conduct a thorough comparative analysis of different eye gaze estimation models, specifically focusing on GazeRefineNet, Odabe, and OpenGaze. By evaluating these models in various application-specific environments, this study seeks to understand their performance in terms of accuracy, computational efficiency, and ease of use.

### Key Objectives
- Assess the performance of each model using the GazeCapture dataset.
- Propose improvements to enhance the precision and operational breadth of these models.
- Identify the most suitable model for specific applications by considering a set of crucial factors including scalability and user accessibility.

## Results

Our comparative analysis yielded significant insights into the strengths and limitations of each model, which are summarized as follows:

### GazeRefineNet
- Demonstrated high accuracy in gaze estimation using standard webcam images, making it a cost-effective solution.
- Achieved a reduction in angular error, suggesting its potential for broad application in user interface accessibility and digital marketing.

### OpenGaze
- Implemented as an open-source model, it offers flexibility and ease of integration for developers.
- Showed comparable accuracy to proprietary systems, validating its effectiveness for educational tools and healthcare applications.

### ODABE
- Featured online transfer learning to adapt quickly to new user environments, significantly reducing prediction errors.
- Its robust performance across diverse settings makes it ideal for real-time applications in dynamic environments.




## Installation

### Prerequisites
Before you begin, ensure you have the following installed:
- Python 3.8+

### Setup
To set up this project locally, run the following commands:
Run on bash
- git clone https://github.com/paro2708/SER517_Group35_Capstone
- cd SER517_Group35_Capstone
- install pytorch modules using the following command (optional)
```
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
- install requirements.txt
- download shape_predictor_68_face_landmarks.dat file form [Github Repository](https://github.com/italojs/facial-landmarks-recognition.git) 

### Data Preprocessing
Before using the main functionalities of this project, it is essential to preprocess the dataset to ensure optimal performance and accuracy of the software. The preprocessing steps are facilitated by several scripts located in the `utils` directory. These scripts are designed to clean, normalize, and prepare the data for analysis or model training.

#### Files in the `utils` Directory
- preProcess.py: This script is responsible for preparing the data before analysis or modeling. It typically includes steps such as cleaning, transforming, and standardizing the data to ensure it is ready for further processing.

- fetchLandmarks.py: This file is used to identify and extract facial landmarks from images. This script typically employs facial recognition algorithms to pinpoint key features on the face, such as the eyes, nose, mouth, and jawline, which are crucial for advanced image processing tasks or facial analysis applications.

- extractEye.py: This script is designed to detect and extract the eye region from images. It likely uses image processing techniques or machine learning models to locate eyes within a given image and isolate them for further analysis or processing.

- normalizeData.py: This script is designed to normalize the data by scaling the features to a standard range. This process ensures that all input features contribute equally to the analysis and prevents any one feature from dominating due to its scale.

#### Running the Preprocessing Scripts

To run the preprocessing scripts, navigate to the `utils` directory and execute the following commands:

```
   cd utils
   python preProcess.py
   python fetchLandmarks.py
   python extractEye.py
   python normalizeData.py
```


#### GazeRefineNet
- GazeRefineNet.py - This file contains the core implementation of the GazeRefineNet model. It defines the neural network architecture used for refining gaze estimates, including layer configurations, forward pass definitions, and loss calculations specific to gaze estimation tasks.

- GazeRefineNetEnhanced.py - This script is an extension or an enhanced version of the GazeRefineNet model. It includes improvements or modifications to the original architecture, such as additional layers, and enhanced training techniques to improve accuracy and performance.

- deviceSeparate.py - This file is responsible for separating the dataset based on the devices that are used by the respective subjects in the dataset. It is used to find dimensions of these devices.

- gazeRefineNetData.py - This script is designed to handle data loading and preprocessing specifically tailored for the GazeRefineNet model. It includes functionality to read gaze data from files, preprocess it (e.g., normalization, augmentation), and organize it into a format suitable for model training and evaluation.

- train.py - As commonly used in machine learning projects, this file contains the training loop for the GazeRefineNet model. It includes loading the model and data, setting up the training parameters (like the optimizer and loss function), running the training epochs, and saving the trained model. It might also include validation/testing within or after the training process to monitor the model's performance.

#### Running the GazeRefineNet Model
Run on bash
```
   cd GazeRefineNet Model
   python train.py
```
#### Opengaze Model
- openGaze.py - This file contains the core implementation of the OpenGaze model. It defines the neural network architecture used for refining gaze estimates, including layer configurations, forward pass definitions, and loss calculations specific to gaze estimation tasks.

- openGazeData.py - This script is designed to handle data loading and preprocessing specifically tailored for the OpenGaze model. It includes functionality to read gaze data from files, preprocess it (e.g., normalization), and organize it into a format suitable for model training and evaluation.

- train.py - As commonly used in machine learning projects, this file contains the training loop for the Opengaze model. It includes loading the model and data, setting up the training parameters, running the training epochs, and saving the trained model. 

- test.ipynb - This is a python notebook provided to help in model testing and evalutions. It will also help in visualization of the results, generating scatterplots. 

#### Running the OpenGaze Model
```
   cd OpenGaze Model
   python train.py
```

#### ODABE Model
- data_load.py - Loads and prepares datasets for training, including the GazeCapture dataset and custom datasets. It provides functions to load GazeCapture dataframes, custom dataframes, and decorate datasets with resizing, normalizing, and conversion to tensors.

- model_vgg.py - Defines a VGG neural network model for gaze estimation. It consists of a feature extractor followed by a regressor for predicting x and y coordinates. The feature extractor is created using VGG architecture, and the regressor consists of fully connected layers with ReLU activation.

- prepare_custom_dataset.py - Prepares custom datasets by resizing images, extracting faces, and generating metadata containing image filenames and corresponding x, y coordinates normalized to the image size.

- prepare_gc_face_gaze.py - Prepares the GazeCapture dataset for gaze estimation by resizing images, extracting faces, and generating metadata containing image filenames and corresponding x, y coordinates normalized to the screen size.

- train.py - Trains the gaze estimation model using the prepared datasets. It includes functions for training, evaluation, and fine-tuning the model. It also handles saving and loading model weights and generating debug images during training for visualization purposes.

#### Running the ODABE Model
```
   cd ODABE Model
   python train.py
```

### Challenges and Future Work
- The study identifies the need for improvements in computational efficiency to enhance the robustness and adaptability of these models.
- Future research will focus on refining these technologies to support a wider array of applications, making eye-tracking more accessible and inclusive.

## Conclusion

This study provides a foundation for future advancements in eye-tracking technology by highlighting effective models and areas for enhancement. The results encourage ongoing development to create more intuitive, engaging, and accessible digital interfaces using eye gaze tracking technology.
