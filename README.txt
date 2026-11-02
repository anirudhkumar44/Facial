FACIAL EMOTION RECOGNITION - CAPSTONE PROJECT
==============================================

PROJECT FILES
-------------
app.py                 Streamlit web application
train_model.py         CNN model training script
class_indices.json     Example emotion class labels
requirements.txt       Required Python packages
emotion_model.keras    Created by train_model.py after training

IMPORTANT ABOUT THE .KERAS MODEL
--------------------------------
A genuine trained .keras model cannot be supplied honestly without training
data and a training run. This package therefore includes the complete CNN
training script. Running it on a labelled dataset creates emotion_model.keras.
Do not treat an untrained/random model as an accurate emotion detector.

DATASET FOLDER
--------------
Create this structure inside this project folder:

dataset/
  angry/
  disgust/
  fear/
  happy/
  neutral/
  sad/
  surprise/

Put correctly labelled face images in each folder. Use a public facial
expression dataset such as FER2013, and check its licence/terms before use.
Keep training, validation and test subjects separated where possible to
reduce data leakage.

INSTALL
-------
python -m venv .venv

Windows:
.venv\Scripts\activate

macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt

TRAIN
-----
python train_model.py

This produces:
- emotion_model.keras
- class_indices.json

RUN STREAMLIT
-------------
streamlit run app.py

MODEL INPUT
-----------
48 x 48 pixels, grayscale, pixel values [0, 255]. The model's internal
Rescaling layer converts them to [0, 1]. The model predicts one of seven labels.

ACADEMIC / ETHICAL NOTE
-----------------------
Facial-expression classification estimates visible patterns in an image.
It cannot reliably determine a person's internal emotional state, intent,
personality, honesty, or mental health. Results can be affected by lighting,
pose, image quality, culture, age, dataset bias and occlusion. Use only
consented images, do not use this demo for consequential decisions, and
report test-set performance and limitations in the capstone report.
