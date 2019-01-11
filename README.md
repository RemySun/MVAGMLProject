# MVAGMLProject

This is the code for the project Graphs embedding for cross model embedding project that was based on the study of the article Learning Semantic Structure-preserved Embeddings for
Cross-modal Retrieval.

## Preprocessing

Preprocessing files exist to create and load processed data files for the PASCALVOC2007 and MIRFLICKR datasets. Images are preprocessed into network features by preprocess\_imgs.py (fill out image paths).

Other data loading functions are contained in utils.py (see train.py script for an example of how to load data).

## Training

The script train.py is configured to run training on PASCALVOC2007, see hyperparameters in the parseArgs() function in utils.py

test.py will provide corresponding quantitative evaluation data in the format (R@K,Medr,precision-scope,MAP@all).
