This file contains the Python implementation of Weighted Similarity Matching Based Biologically-Plausible Determinant Maximization Neural Networks for Blind Source Separation. The following is the contents of the subfolders.

-reqirements.txt : Python library requirements.

>src : Contains the WSMBSS algorithm and baseline algorithms LD-InfoMax, BSM, NSM, ICA as well as utility functions. There are 4 python scripts : WSMBSS.py, WSMBSSv2.py, LDMIBSS.py, and utils.py

>Numerical Experiments:
>> BaselineTutorialNotebooks : Contains several example experiment notebooks (in .ipynb format) for LD-InfoMax, BSM and NSM. For ICA usage, check ImageSeparation folder. As long as your python environment include the libraries from requirements.txt (also you need jupyter-notebook or jupyter-lab), these notebooks can be executed.
>> ImageSeparation : Contains the image separation experiment in the paper. One notebook includes the experiment for WSM, and the other notebooks include results for LD-InfoMax, NSM and ICA. Subfolder TestImages include the RGB images used in the experiments.
>> WSMTutorialNotebooks : Contains multiple example source separation experiments for anti-sparse, nonnegative anti-sparse, sparse, nonnegative-sparse, mixed anti-sparse, and other identifiable domains such as given experiment in Polytopic Matrix Factorization (PMF). As long as your python environment include the libraries from requirements.txt (also you need jupyter-notebook or jupyter-lab), these notebooks can be executed.
>> SimulationForPaper : Contains the Python scripts for the experiments presented in the paper manuscript and appendix. These scripts are the extensions of the codes presented in the WSMTutorialNotebooks for multiple runs to measure average convergence behavior. Each subfolder name indicates the source domain of the executed experiment. Please do not forget to include corresponding WSMBSS script from src folder to run the experiments (check import lines).
>> 4PAMExample: This file contains the simulation codes for digital communication example in the paper (Section E.8) with 4PAM scheme.
