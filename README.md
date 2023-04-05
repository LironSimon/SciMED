# SciMED: A Computational Framework For Physics-Informed Symbolic Regression with Scientist-In-The-Loop

## Abstract
Discovering a meaningful, dimensionally homogeneous, symbolic expression explaining
experimental data is a fundamental challenge in physics. In this study, we present a
novel, open-source computational framework called Scientist-Machine Equation
Detector (SciMED), which integrates scientific discipline wisdom in a
scientist-in-the-loop approach with state-of-the-art SR methods. SciMED combines a
genetic algorithm-based wrapper selection method with automatic machine learning
and two levels of symbolic regression methods. We test SciMED on four configurations
of the settling of a sphere with and without a non-linear aerodynamic drag force. We
show that SciMED is sufficiently robust to discover the correct physically meaningful
symbolic expressions of each configuration from noisy data. Our results indicate better
performance on these tasks than the state-of-the-art SR software package.

## Table of contents
1. [Code usage](#code_usage)
2. [The algorithm](#the_algorithm)
3. [Data](#data_preparation)
4. [How to cite](#how_to_cite)
5. [Dependencies](#dependencies)
6. [Contributing](#contributing)
7. [Bug Reports](#bug_reports)
8. [Contact](#contact)

<a name="code_usage"/>

## Code usage
### Run the experiments shown in the paper:
1. Clone the repo 
2. Install the `requirements.txt` file.
3. run the project from the `paper_exp_runner.py` file, make sure all the arguments are set to **True**. 

### Use in your project:
1. Clone the repo 
2. Install the `requirements.txt` file.
3. Include the following code to the relevant part of your project:
```
from scimed import scimed
scimed.run(dataset_x: pandas.DataFrame, dataset_y: pandas.Seires, ...)
```
### Demo:
A demo on how to use SciMED with a data from a CSV file (using Pandas) is shown in the "/demo" folder. 

<a name="the_algorithm"/>

## The algorithm
SciMED is constructed from four components: 
1. **A genetic algorithm-based feature selection:** Reduce the search space by selecting a single, most explainable, feature from each group of features that are considered to be the same in physical essence. This devition to groups is provided by the user, applying his domain knowledge.
2. **A genetic algorithm-based automatic machine learning (AutoML):** Trains an ML to produce synthetic data that facilitates the SR task by enriching the data domain.
3. **A genetic algorithm-based symbolic regression (SR):** less resource and time-consuming but stochastic SR search. May result in sub-optimal outcome.
4. **A Las Vegas search SR:** more computationally expensive SR search that averagely produces more stable and accurate outcome.

Each section allows the user to easily insert physical knowledge or assumptions, specific to its current task, directing the search process for a more credible result 
with fewer required resources. The motivation for this structure is derived from the way human scientists work, where more promising directions get more attention and resources. 

![Algo_structure](https://user-images.githubusercontent.com/72650415/189652380-9a3104d8-dd12-4629-9814-7ae6774babdb.png)

<a name="data_preparation"/>

## Data preparation
The data file to be analyzed should be a csv file, with each column containing the numerical values of each variable. If the variables can be grouped into variables of similar essence, from which only one can be in the mystery eqaution, then they should appear sequentially and the index ranges for each group should be passed to the function.

The solution file will be saved in the directory called "results" under the name of the specific component that generated them. For example, there will be three {component}_target_vs_pred.pdf files demonstrating the prediction capabilities of the specific outcome from the component. 

<a name="how_to_cite"/>

## How to cite
Please cite the SciMED work if you compare, use, or build on it:
```
@article{keren2023computational,
        title={A computational framework for physics-informed symbolic regression with straightforward integration of domain knowledge},
        author={Keren, Liron Simon and Liberzon, Alex and Lazebnik, Teddy},
        journal={Scientific Reports},
        volume={13},
        number={1},
        pages={1249},
        year={2023},
        publisher={Nature Publishing Group UK London}
}
```

<a name="dependencies"/>

## Dependencies 
1. pandas 
2. numpy 
3. matplotlib 
4. seaborn 
5. scikit-learn 
6. scipy 
7. TPOT 
8. gplearn 
9. pytorch 
10. termcolor 
11. sympy

<a name="contributing"/>

## Contributing
We would love you to contribute to this project, pull requests are very welcome! Please send us an email with your suggestions or requests...

<a name="bug_reports"/>

## Bug Reports
Report [here]("https://github.com/LironSimon/SciMED/issues"). Guaranteed reply as fast as we can :)

<a name="contact"/>

## Contact
* Liron Simon - [email](mailto:lirons.gb@gmail.com) | [LinkedInֿ](https://www.linkedin.com/in/liron-simon/)
* Teddy Lazebnik - [email](mailto:t.lazebnik@ucl.ac.uk) | [LinkedInֿ](https://www.linkedin.com/in/teddy-lazebnik/)
* Alex Liberzon - [email](mailto:alexlib@tauex.tau.ac.il) | [LinkedInֿ](https://www.linkedin.com/in/alexliberzon/)


## Run online using Mybinder.org

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/LironSimon/SciMED/master)

Open New Terminal and run `python main.py`


## Run using Docker

    docker run alexlib/scimed:latest
 

