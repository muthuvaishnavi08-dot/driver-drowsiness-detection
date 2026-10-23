# Driver Drowsiness Detection

## Overview
A computer vision project for detecting driver drowsiness using facial features, machine learning, and deep learning.

## Features
- Facial feature extraction using MediaPipe and OpenCV
- Eye Aspect Ratio (EAR) and mouth-opening analysis
- Machine-learning models: Random Forest, XGBoost, and SVM
- Deep-learning models: CNN and CNN-LSTM
- Webcam-based detection prototype
- Audio alerts for drowsiness

## Technologies Used
- Python
- OpenCV
- MediaPipe
- NumPy and Pandas
- Scikit-learn
- XGBoost
- TensorFlow / Keras

## Project Structure
- `main.py` — Main webcam application
- `drowsiness_detector.py` — Alternative detection implementation
- `Python drowsiness .ipynb` — Model training and experiments
- `driver_drowsiness_features_1000.csv` — Extracted feature samples
- `audio_alert.py` — Audio alert utilities

## Setup
1. Install Python and create a virtual environment.
2. Install the required dependencies.
3. Configure the dataset path in the notebook.
4. Run the notebook to explore the model experiments.
5. Run `main.py` to launch the webcam prototype.

## Results
The notebook reports preliminary image-classification results of approximately 99% accuracy for the CNN and 99.3% for the CNN-LSTM. These results have not been independently validated on subject-independent test data.

## Limitations
The project is a prototype and is not validated for safety-critical driving use.

## Future Improvements
- Improve facial landmark and eye-closure detection.
- Train and evaluate on subject-independent data.
- Use consecutive video frames for temporal modelling.
- Improve real-time alarm handling and robustness under different lighting conditions.
