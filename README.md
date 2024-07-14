# ML

## Versions

Currently v.0.2. Refer to CHANGELOG.md for a details.

## Description

Scripts to streamline the ML fitting process. Currently focuses mainly on multiple linear regression applied from a Design of Experiments perspective.

## Getting Started

### Dependencies

Uses common ML/DS packages like pandas, matplotlib, statsmodels, scikit-learn, etc. To install all the dependencies, run 

```
pip install -r requirements.txt
```

### Downloading a copy

The script can be either manually downlaoded from Github or installed via Git.

#### Downloading from Github

* Go to [the Github repo](https://github.com/mfvhidalgo/ML)
* Hit the green <u>**<> Code**</u> button on the upper right corner then hit <u>**Download ZIP**</u>.
* The files of interest are in the src folder

#### Downloading using Git

* open Git bash
* enter
```
git clone https://github.com/mfvhidalgo/ML
```

## Running the scripts

Data should first be entered into Data.xlsx. Afterwards, specific ML scripts can be run.

### Entering data

* Data is entered through src\Data.xlsx.
* See a sample in \docs\examples and guides\input data\Data.xlsx.

### Multiple Linear Regression (MLR)

* Open src\auto_mlr.py.
* Enter the list of terms to be used in the MLR model in terms_list. Each term follows the [patsy](https://patsy.readthedocs.io/en/latest/) convention.

## Upcoming features

### Features

* implement logistic regression
* implement RF, XGBoost, NN

### Guides

* guide for using Data.xlsx.
* guide for using auto_mlr
* guide for using functions\mult_lin_reg_utils

## Authors

Marc Francis V. Hidalgo, PhD
    [[Github](https://github.com/mfvhidalgo/)]
    [[LinkedIn](https://www.linkedin.com/in/mfvhidalgo/)]

## License

This project is licensed under the MIT License - see the LICENSE.md file for details