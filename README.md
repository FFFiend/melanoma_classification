# Interactive Melanoma Classifier (Demo [here](https://melanomas.streamlit.app))
Training and Validation/Testing accuracies on the dataset:
![Training loss](train.png)
![Test/Validation](accuracy.png)

# Update: 
Latest model architecture (model.py) showing promising results (>= 80% on testing set),
uploading model.pkl soon.
# current TODO
2) pickle dump file 

## Pickle dump model from Colab to pkl file
with open('model_pkl', 'wb') as file:
    pickle.dump(model, file)
###

