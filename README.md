# Interactive Melanoma Classifier (Demo [here](https://melanomas.streamlit.app))

# Update: 
Taking longer than expected as current arch and hyperparams give around 60% validation acc :/

I will be uploading the .ipynb and model.pkl because I'm gonna end up training the model on cloud, so perhaps the Python scripts themselves (except host.py) will be deprecated soon.

# current TODO
1) train on colab (IPR)
2) pickle dump file 

## Pickle dump model from Colab to pkl file
with open('model_pkl', 'wb') as file:
    pickle.dump(model, file)
###

