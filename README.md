<!-- 
This README file provides an overview of the repository, which includes the source code and results of an empirical study. 
The study is described in the paper titled "Robustness Evaluation of Counterfactual Explanations from Generative Models: An Empirical Study". 
-->
This repository contains the source code and results of empirical study described in the paper "Robustness Evaluation of Counterfactual Explanations from Generative Models: An Empirical Study". 

# Structure
- ```datasets```
- ```models```
- ```cf_methods```
- ```utils```

# Usage
In order to use ```src``` as a module for running experiments on evaluation of the present generative models that produce counterfactual explanations, one is required to execute  ```pip install -e .``` in the folder where the ```setup.py``` file is located. 

# CF generation models
In this project, we examined the following models:

1) REVISE [1]
2) CounteRGAN [2]
3) C3LT [3]

# *References*

[1] - Joshi, S., Koyejo, O., Vijitbenjaronk, W., Kim, B., & Ghosh, J. (2019). Towards realistic individual recourse and actionable explanations in black-box decision making systems. arXiv preprint arXiv:1907.09615.
[2] - Nemirovsky, D., Thiebaut, N., Xu, Y., & Gupta, A. (2022, August). Countergan: Generating counterfactuals for real-time recourse and interpretability using residual gans. In Uncertainty in Artificial Intelligence (pp. 1488-1497). PMLR.
[3] - Khorram, S., & Fuxin, L. (2022). Cycle-consistent counterfactuals by latent transformations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 10203-10212).


