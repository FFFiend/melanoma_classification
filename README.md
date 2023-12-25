# Interactive Melanoma Classifier (Completed & Deployed [here](https://melanomas.streamlit.app))
This is a classifier based on a CNN model that I wrote myself, achieves around 80-85% accuracy on the 
following [dataset](https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images/data).

As a result, the model works best on images pulled directly from the dataset's testing set, and the next step is to 
try transfer learning to increase accuracy on a wider range of images.

The entire training process (epoch #, batch sizes, etc) can be seen inside the .ipynb file in the training folder, the architecture itself is in ```model.py``` and my trained weights are in ```model```. Feel free to look around.

Here are the Training and Validation/Testing accuracies on the dataset:
![Training loss](assets/train.png)
![Test/Validation](assets/accuracy.png)

# Improvements
Use max pooling on larger inputs instead of cropping from center and padding for images smaller than 300x300.

