# Interactive Melanoma Classifier (Demo [here](https://melanomas.streamlit.app))

# Update: 
Taking longer than expected as current arch and hyperparams give around 60% validation acc :/
Will upload model.pkl file when I've achieved > 85% accuracy
# current TODO
1) train on colab (IPR)
2) pickle dump file 

## Pickle dump model from Colab to pkl file
with open('model_pkl', 'wb') as file:
    pickle.dump(model, file)
###

