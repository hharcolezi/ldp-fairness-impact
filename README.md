## LDP Impact on Fairness
Repository for the paper: *Héber H. Arcolezi, Karima Makhlouf, and Catuscia Palamidessi. "[(Local) Differential Privacy has NO Disparate Impact on Fairness](https://arxiv.org/abs/2304.12845)" arXiv preprint arXiv:2304.12845 (2023).* Accepted at [DBSec 2023](http://www.dbsec2023.unimol.it/).

If our codes and work are useful to you, we would appreciate a reference to:
```
@article{arcolezi2023local,
  title={(Local) Differential Privacy has NO Disparate Impact on Fairness},
  author={Arcolezi, H{\'e}ber H and Makhlouf, Karima and Palamidessi, Catuscia},
  journal={arXiv preprint arXiv:2304.12845},
  year={2023}
}
```

## Environment
Our codes were developed using Python 3 with mainly numpy, pandas, lightgbm, and multi-freq-ldpy libraries. The versions we used are listed below:

- Python 3.9.13
- Numpy 1.21.5
- Pandas 1.4.4
- Lightgbm 3.3.5
- Multi-freq-ldpy 0.2.4

## Organization of Experiments
This repository is organized/ordered in several Jupyter Notebook files as:
- `0_Preprocess_Datasets.ipynb:` Notebook for pre-processing (cleaning, encoding) three original datasets (i.e., Adult, ACSCoverage, LSAC). The pre-processed datasets are saved in the `datasets` folder.
- `1_BO_NonDP.ipynb:` Notebook for conducting the Bayesian Optimization (BO) to find the local optimal LGBM hyperparameters using original data (i.e., no LDP). Results are saved in CSV format in the `results` folder.
- `2_Exp_XXX:` Notebooks for carrying out all experiments (repeated over `nb_seed=20` iterations) of the paper assuming the mechanism `XXX` (e.g., NonDP as the baseline, GRR, OUE, etc). Results are saved in CSV and numpy formats in the corresponding `results` folder. The notebook `2_Exp_All_LDP` is generic and runs all LDP protocols. 
- `3_Final_Results.ipynb:` Notebook with codes to plot the final results illustrated in the main paper.
- `4_Appendix_Experiments.ipynb`: Notebook for carrying out the additional set of experiments (repeated over `nb_seed=20` iterations) with all LDP protocols. Results are also saved in CSV format in the `results` folder. 
- `4_Appendix_Results.ipynb`: Notebook with codes to plot the final results illustrated in the appendix of the full paper (in arXiv).


Some functions used in the Jupyter Notebooks are imported from `functions.py`.

## To Do
We are slowly cleaning/generalizing the codes + documentation.

## Acknowledgements
- We use the LDP protocols implemented in our [multi-freq-ldpy](https://github.com/hharcolezi/multi-freq-ldpy) library.
- We use the reconstructed Adult and ACSCoverage datasets from the [folktables](https://github.com/socialfoundations/folktables) library.
- We use the [LSAC](https://eric.ed.gov/?id=ED469370) dataset. 

## Contact
For any questions, please contact: 
- [Héber H. Arcolezi](https://hharcolezi.github.io/): heber.hwang-arcolezi [at] inria.fr
- [Karima Makhlouf](http://www.lix.polytechnique.fr/Labo/Karima.MAKHLOUF/): karima.makhlouf [at] lix.polytechnique.fr

## License
[MIT](https://github.com/hharcolezi/ldp-fairness-impact/blob/main/LICENSE)