# melanoma_classification
~~Melanoma classification full stack site, model written and trained from scratch. MongoDB, Express/Node, React and Pytorch/JStorch/Jax for the model. Hosting TBD~~

# Update: 
switching to Streamlit for deployment instead.

# current TODO
1) finish arch
2) train on colab
3) pickle dump file 
4) add callback on predict button, done.

## Pickle dump model from Colab to pkl file
with open('model_pkl', 'wb') as file:
    pickle.dump(model, file)
###

