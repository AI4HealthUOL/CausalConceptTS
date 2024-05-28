# CausalConceptTS: Causal Attributions for Time Series Classification using High Fidelity Diffusion Models


This is the official repository for the paper CausalConceptTS: Causal Attributions for Time Series Classification using High Fidelity Diffusion Models

[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](https://arxiv.org/abs/2405.15871)

In this study, within the context of time series classification, we introduce a novel framework to assess the causal effect of concepts, i.e., predefined segments within a time series, on specific classification outcomes. To achieve this, we leverage state-of-the-art diffusion-based generative models to estimate counterfactual outcomes.


### Results 

We prove our approach efficace through three tasks: 

- Drought prediction 

[alt-text-1][reports/ptbxl_concepts.pdf] ![alt-text-2](image2.png "title-2")

- ECG classification

![alt-text-1](image1.png "title-1") ![alt-text-2](image2.png "title-2")

- EEG classification

![alt-text-1](image1.png "title-1") ![alt-text-2](image2.png "title-2")



### Experiments

- Download the data from [this link](https://mega.nz/folder/5aVXyQqA#WmpdfRZVVsXM3x_xnDo1TA)

- Place the desired test set under the data directory

- Follow the instructions under demo.ipynb to obtain the causal effects.


#### We welcome contributions to improve the reproducibility of this project! Feel free to submit pull requests or open issues.


## Reference
```bibtex
@misc{alcaraz2024causalconceptts,
      title={CausalConceptTS: Causal Attributions for Time Series Classification using High Fidelity Diffusion Models}, 
      author={Juan Miguel Lopez Alcaraz and Nils Strodthoff},
      year={2024},
      eprint={2405.15871},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
