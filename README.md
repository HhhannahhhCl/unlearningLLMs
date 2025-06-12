# Parameter-Efficient Unlearning for Large Language Models – Leveraging NPO, Rényi Divergence, and Sequential Data Sampling

This GitHub repository contains the code to the paper "Parameter-Efficient Unlearning for Large Language Models – Leveraging NPO, Rényi Divergence, and Sequential Data Sampling" by Clausen (2025) implementing an unlearning algorithm combining previous work by Premptis et al. (2025) and Bronec and Helcl (2025). The system uses sequential data sampling during training as well as implements a loss function based on NPO, cross entropy and the Rényi divergence. 

The best model trained and evaluated on the validation data can be found on huggingface: [hannacla/unlearned_OLMo-7B](https://huggingface.co/hannacla/unlearned_OLMo-7B)

In order to test the algorithm itself, the finetuning.slurm file may be executed. The files utils.py, data.py and sequential.py contain the necessary functions. The default values will finetune the OLMo 1B model with all loss terms scaled equally as well as alpha and beta set to 0.5. Note that an output directory needs to be specified to save data on the training and the trained model. In order to run the evaluation, first the mmlu_eval.slurm script needs to be executed by providing the path to the model, an output path and a batch size. Afterwards, the model can be evaluated by running the unlearn_evaluation.slurm script by providing the path to the evaluation data, the model, the mia data, the mmlu evaluation results, the maximum number of new tokens, the batch size as well as a parameter to define a concrete output path. Finally, executing the manual.slurm script with default values will provide a sample of generated output for a qualitative analysis.

### References

Jan Bronec and Jindřich Helcl. 2025. [Atyaephyra at SemEval-2025 Task 4: Low-Rank NPO](https://arxiv.org/abs/2503.13690). Preprint, arXiv:2503.13690.

Iraklis Premptis, Maria Lymperaiou, Giorgos Filandrianos, Orfeas Menis Mastromichalakis, Athanasios Voulodimos, and Giorgos Stamou. 2025. [AILS-NTUA at SemEval-2025 Task 4: Parameter-Efficient Unlearning for Large Language Models using Data Chunking](https://arxiv.org/abs/2503.02443). Preprint, arXiv:2503.02443.

### Citation
```
@inproceedings{clausen-2025-unlearning,
	title = "Parameter-Efficient Unlearning for Large Language Models – Leveraging NPO, Rényi Divergence, and Sequential Data Sampling",
	author = "Hannah Clausen",
	booktitle = "Proceedings of the Seventh IN5550 Workshop on Neural Natural Language Processing (WNNLP 2025)",
	editors = "Andrey Kutuzov and David Samuel and Vladislav Mikhailov and Roxana Pop and Sondre Wold",
	month = june,
	year = "2025",
	publisher = "University of Oslo, Norway",
	pages = "88--98"
}
```
