# CausalConceptTS: Causal Attributions for Time Series Classification using High Fidelity Diffusion Models


This is the official repository for the paper CausalConceptTS: Causal Attributions for Time Series Classification using High Fidelity Diffusion Models


In this study, within the context of time series classification, we introduce a novel framework to assess the causal effect of concepts, i.e., predefined segments within a time series, on specific classification outcomes. To achieve this, we leverage state-of-the-art diffusion-based generative models to estimate counterfactual outcomes.


### Results 

We prove our approach efficace through three tasks: 

- Drought prediction 

![alt text](https://github.com/AI4HealthUOL/CausalConceptTS/blob/main/reports/drought_concepts_learned.png?style=centerme)
![alt text](https://github.com/AI4HealthUOL/CausalConceptTS/blob/main/reports/drought_do1_do(causal).png?style=centerme)

- ECG classification

![alt text](https://github.com/AI4HealthUOL/CausalConceptTS/blob/main/reports/ptbxl_concepts.png?style=centerme)
![alt text](https://github.com/AI4HealthUOL/CausalConceptTS/blob/main/reports/ptbxl_do1_do(causal).png?style=centerme)

- EEG classification

![alt text](https://github.com/AI4HealthUOL/CausalConceptTS/blob/main/reports/schizo_concepts.png?style=centerme)
![alt text](https://github.com/AI4HealthUOL/CausalConceptTS/blob/main/reports/schizo_do1_do(causal).png?style=centerme)


### Experiments

- Download the data from [this link](https://mega.nz/folder/5aVXyQqA#WmpdfRZVVsXM3x_xnDo1TA)

- Place the desired test set under the data directory

- Follow the instructions under demo.ipynb to obtain the causal effects.


#### We welcome contributions to improve the reproducibility of this project! Feel free to submit pull requests or open issues.

