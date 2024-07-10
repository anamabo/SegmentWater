# SegmentWater
Segmentation of water bodies using PaliGemma: Google's Visual Language Model.

## General set up and activation of the environment 
If you become developer of this project, clone this repository first.

Once the project is cloned, you need to create a virtual environment. To do so,  
open a terminal and type the following:

```
$ pipenv install --dev
```

To  activate the environment, type:

```
$ pipenv shell
```

We want to maintain high quality standards in our code. For this, we use `pre-commit`. To install it 
in your virtual environment, type:

```
$ pre-commit install 
```

This plugin will facilitate and automate the code formatting.

## Install new packages 
If you need to install new PyPI packages, run this command:
```
$ pipenv install <package-name>
```
For more options visit the [pipenv official website](https://pipenv.pypa.io/en/latest/installation/#installing-packages-for-your-project).


## Naming convention for the Jupyter Notebooks

To ease collaborative development and versioning of Jupyter notebooks, we are using the following naming convention:
`number_author'sname_description.ipynb`. For instance, a notebook developed by Carmen that has some EDA on data, can
have the name `1.CAMB_EDA_enriched_dataset.ipynb`.
