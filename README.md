# A topic model for the debates of the 18th German Bundestag as a showcase example 

Markus Konrad <markus.konrad@wzb.eu>, May 2018

## Overview

For a workshop on practical topic modeling, I created this topic model as a showcase example that demostrates the steps that are necessary to take in order to arrive at a usable, informative model:

1. Preprocessing the raw data (`preproc_raw.py`)
2. Generating the document-term-matrix from the data (`generate_tokens.py`)
3. Evaluating topic models for a set of hyperparameters (`tm_eval.py` and `tm_eval_plot.py`)
4. Generating the final model using the best combination of hyperparameters (`generate_model.py`)
5. Visualizing, interpreting and analysing the model (`report1.ipynb`, `report2.ipynb` and `example_analyses.py`) – note that this was not the focus of the workshop and hence only exemplary analyses are given
   

## Used software packages

This example uses Python 2.7 because of some dependency issues (namely the [pattern package](https://github.com/clips/pattern) for better lemmatization of German texts does not support Python 3).

These are the main software packages in use:

* [tmtoolkit](https://github.com/WZBSocialScienceCenter/tmtoolkit) for evaluating models in parallel, calculating some model statistics and visualizations
* [lda](https://github.com/lda-project/lda) for topic modeling with LDA using Gibbs sampling
* [PyLDAVis](https://pyldavis.readthedocs.io/en/latest/) and [Jupyter Notebooks](https://jupyter.org/) for interactive visualizations

All software dependencies can be installed via `pip install -r requirements.txt`.

## Data

The data for the debates comes from [offenesparlament.de](https://github.com/Datenschule/offenesparlament-data).

## License

Licensed under [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) (except for the data from offenesparlament.de which have their own license). See `LICENSE` file. 
