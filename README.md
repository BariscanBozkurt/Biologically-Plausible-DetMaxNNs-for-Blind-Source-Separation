# Biologically-Plausible Determinant Maximization Neural Networks for Blind Separation of Correlated Sources

This Github repository includes the implementation of the proposed approaches presented in the [paper](https://arxiv.org/abs/2209.12894), which is accepted at NeurIPS 2022.

## DetMax Neural Networks

General source domain with sparse components            |  Antisparse sources
:-------------------------:|:-------------------------:
![Sample Network Figures](./Figures/networkfigurenewsqueezed1.png)   |  ![Sample Network Figures](./Figures/NNantisparsesqueezed1.png)

## To Run Simulations

The folder "Simulations" includes the codes for example experiments. The subfolders are named accordingly, e.g., "SparseNoisy" folder contains the experiments for the sparse source separation simulations. The jupyter notebooks inside the folder "Simulations/AnalyzeSimulationResults" illustrates the plots and tables of the experiment results. For example, the notebook "PlotSimulationResults_SparseNoisy.ipynb" includes the plots for the sparse source separation experiments. To replicate the figures in this specific notebook, you need to follow the below steps,

 * Run the python script in the folder "Simulations/SparseNoisy" with the following command:

    ``` python WSM_Sparse_NoisyV1.py```

 * When you run the python simulation, the following pickle file will be created which contains the SINR results.

    * "Simulations/Results/simulation_results_sparse_noisyV1.pkl"

 * The jupyter notebook "Simulations/AnalyzeSimulationResultsFinal/PlotSimulationResults_SparseNoisy.ipynb" reads the above pickle file and visualize the SINR convergence results. Moreover, the performances of the batch (or baseline) algorithms are also reported.

To replicate simulations in the given folders, you can adapt the above procedure for the other examples (i.e., antisparse blind source separation (BSS) simulations). The experiment for image separation is included in "Simulations/ImageSeparation" as Jupyter Notebook files. The sparse dictionary learning experiment is located in "Simulations/SparseDictionaryLearning", and the notebook inside this folder produces the sparse receptive fields from prewhitened Olshaussen's image pathces (you need "imagepatcheselfwhitened.mat" file whereas it is not included here due to its file size).


## Python Version and Dependencies

* Python Version : Python 3.8.12

* pip version : 21.2.4

* Required Python Packages : Specified in requirements.txt file.

* Platform Info : OS: Linux (x86_64-pc-linux-gnu) CPU: Intel(R) Xeon(R) Gold 6248 CPU @ 2.50GHz

## Folder Contents

### src
This file is the source code for each BSS algorithm we used in the paper. The following is the full list.

Python Script         |  Explanation
:--------------------:|:-------------------------:
BSSbase.py            | Base class for blind source separation algorithms
WSMBSS.py             | Our proposed Weighted similarity mathcing-based (WSM) determinant maximization neural networks for blind separation of correlated sources
BSMBSS.py             | Implementation of Bounded similarity matching (BSM) for uncorrelated antisparse sources [1]
NSMBSS.py             | Implementation of Nonnegative similarity matching (NSM) for uncorrelated nonnegative sources [2]
LDMIBSS.py            | Implementation of Log-det (LD-) Mutual Information maximization (LD-InfoMax) framework for blind separation of correlated sources [3]
PMF.py                | Implementation of Polytopic Matrix Factorization [4]
ICA.py                | Implementation of several independent component analysis frameworks 
bss_utils.py          | Utility functions for blind source separation experiments
dsp_utils.py          | Utility functions for digital signal processing
polytope_utils.py     | Utility functions for polytope operations
visualization_utils.py| Utility functions for visualizations
numba_utils.py        | Utility functions using numba library of Python
general_utils.py      | Other utility functions

### Notebook_Examples
This file includes the jupyter notebook experiments of the algorithms Online WSM, LD-InfoMax, PMF, BSM, NSM, and ICA. The subfolder names are given accordingly, and notebooks are presented to experiment and visualize BSS settings with different algorithms. These notebooks can also be used for debugging and tutorials.

### Simulations


Simulation Folder                     |  Explanation
:------------------------------------:|:-------------------------:
AntisparseCorrelated                  | (Signed) antisparse source separation simulations
NonnegativeAntisparseCorrelated       | Nonnegative antisparse source separation simulations
SparseNoisy                           | Sparse source separation simulations
NonnegativeSparseNoisy                | Nonnegative sparse source separation simulations
SparseDictionaryLearning              | Sparse dictionary learning experiment
SpecialPolytope                       | A BSS simulation on a 3-dimensional identifiable polytope presented in the paper
ImageSeparation                       | Image separation demo
4PAM_Example                          | A BSS simulation setting with 4PAM digital communication signals
AblationStudies                       | Ablation studies on hyperparameter selections
AnalyzeSimulationResults              | Producing the plots and tables for simulation results

## References

[1] Alper T. Erdogan and Cengiz Pehlevan. Blind bounded source separation using neural networks
with local learning rules. In ICASSP 2020 - 2020 IEEE International Conference on Acoustics,
Speech and Signal Processing (ICASSP), pp. 3812–3816, 2020. doi: 10.1109/ICASSP40776.
2020.9053114.

[2] Cengiz Pehlevan, Sreyas Mohan, and Dmitri B Chklovskii. Blind nonnegative source separation
using biological neural networks. Neural computation, 29(11):2925–2954, 2017

[3] Alper T. Erdogan. An information maximization based blind source separation approach for dependent and independent sources. In ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 4378–4382, 2022. doi: 10.1109/ICASSP43922.
2022.9746099.

[4] Gokcan Tatli and Alper T. Erdogan. Polytopic matrix factorization: Determinant maximization
based criterion and identifiability. IEEE Transactions on Signal Processing, 69:5431–5447, 2021.
doi: 10.1109/TSP.2021.3112918.

## Citation
If you find this useful, please cite:
```
@conference {WSMDetMax,
	title = {Biologically-Plausible Determinant Maximization Neural Networks for Blind Separation of Correlated Sources},
	booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
	year = {2022},
	author = {Bariscan Bozkurt and Cengiz Pehlevan and Alper Erdogan}
}
```