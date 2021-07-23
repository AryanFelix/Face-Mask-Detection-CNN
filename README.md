# Face Mask Detector
A Face Mask Detection tool that detects whether the user is wearing a Mask, not wearing a Mask or if the user is wearing the mask incorrectly. The classification model is based on CNN and was made using Keras library. Video and Image capture and modifications are done using OpenCV and PIL libraries.

## Model Metrics
* Accuracy: 98%
* Validation Accuracy: 94%
* Training Loss: 5%

## Steps to Run
1. Obtain the dataset from the link below.
2. Train the model by running all the cells in `FaceTraining.ipynb`.
3. Remember to change the path of the dataset accoring to your liking.
4. Either run `FaceMaskDetection.ipynb` or `FaceMaskDetection.py`. Jupyter notebook gave me an error for my version of OpenCV. It that happens, just run the Python script instead.

## Dataset
* Face Mask Detection Dataset
    
    Masks play a crucial role in protecting the health of individuals against respiratory diseases, as is one of the few precautions available for COVID-19 in the absence of immunization. With this dataset, it is possible to create a model to detect people wearing masks, not wearing them, or wearing masks improperly.This dataset contains 853 images belonging to the 3 classes, as well as their bounding boxes in the PASCAL VOC format.
    
    https://www.kaggle.com/andrewmvd/face-mask-detection


## License
* MIT License

    Copyright © [Aryan Felix](https://github.com/AryanFelix)