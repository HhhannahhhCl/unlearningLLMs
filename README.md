# Parameter-Efficient Unlearning for Large Language Models – An Exploration of New Approaches Leveraging NPO, Rényi Divergence, and Sequential Data Sampling

This GitHub repository contains the code to the respective submitted paper implementing an unlearning algorithm combining previous work by Premptis et al. (2025) and Bronec and Helcl (2025). The system uses sequential data sampling during training as well as implements a loss function based on NPO, cross entropy and the Rényi divergence. While this is an anonymized repository, the identifiable repository has been made available to the examinors. 

Based on the validation data, the best model is located on Fox and will be published after the anonymous peer-review has ended. 

In order to test the algorithm itself, the finetuning.slurm file may be executed. The files utils.py, data.py and sequential.py contain the necessary functions. The default values will finetune the OLMo 1B model with all loss terms scaled equally as well as alpha and beta set to 0.5. In order to run the evaluation, first the mmlu_eval.slurm script needs to be executed by providing the path to the model, an output path and a batch size. Afterwards, the model can be evaluated by running the unlearn_evaluation.slurm script by providing the path to the evaluation data, the model, the mia data, the mmlu evaluation results, the maximum number of new tokens, the batch size as well as a **newly added parameter to define a concrete output path**. Finally, executing the manual.slurm script while providing a path to the model will provide a sample of generated output for a qualitative analysis.
Note that some arguments need to be replaced when executing the different scripts due to the anonymization of path names.

References:

Jan Bronec and Jindřich Helcl. 2025. [Atyaephyra at SemEval-2025 Task 4: Low-Rank NPO](https://arxiv.org/abs/2503.13690). Preprint, arXiv:2503.13690.

Iraklis Premptis, Maria Lymperaiou, Giorgos Filandrianos, Orfeas Menis Mastromichalakis, Athanasios Voulodimos, and Giorgos Stamou. 2025. [AILS-NTUA at SemEval-2025 Task 4: Parameter-Efficient Unlearning for Large Language Models using Data Chunking](https://arxiv.org/abs/2503.02443). Preprint, arXiv:2503.02443.
