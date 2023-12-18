# melanoma_classification
~~Melanoma classification full stack site, model written and trained from scratch. MongoDB, Express/Node, React and Pytorch/JStorch/Jax for the model. Hosting TBD~~

# Finished Deliverable Date: Dec 24 (after exams :()

# Update: 
switching to Streamlit for deployment instead.
I will be uploading the .ipynb and model.pkl because I'm gonna end up training the model on cloud, so perhaps the Python scripts themselves (except host.py) will be deprecated soon.

# current TODO
1) finish arch
2) train on colab
3) pickle dump file 
~~4) add callback on predict button, done.~~

## Pickle dump model from Colab to pkl file
with open('model_pkl', 'wb') as file:
    pickle.dump(model, file)
###

