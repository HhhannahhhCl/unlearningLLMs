import pandas as pd
from datasets import DatasetDict, Dataset, concatenate_datasets


class DatasetProcessor:
    def __init__(self, data_dir, tokenizer, n_samples_per_task=None, start_idx=0, gold_standard=False):
        """
        Initializes the DatasetProcessor with the specified data path, tokenizer, and sampling options.

        Args:
            data_path (str): Path to the dataset files.
            tokenizer: Tokenizer to use for tokenization and masking.
            num_samples (int, optional): Number of samples to select from each task/dataset. Defaults to None (no sampling).
            start_idx (int, optional): Starting index for sampling. Defaults to 0.
        """
        self.data_path = data_dir # /fp/projects01/ec403/IN5550/exam/unlearning
        self.tokenizer = tokenizer
        self.num_samples = n_samples_per_task
        self.start_idx = start_idx
        self.gold_standard = gold_standard

    def create_datasets(self, split, task, split_tasks=True, split_retain=False):
        """
        Creates and returns a DatasetDict from parquet files at the specified data path.

        Args:
            split (str): Split name (e.g., "train", "test").
            task (str): Subtask name ("all", "Task1", "Task2", "Task3"), Defaults to all
            split_tasks (bool, optional): Whether to split datasets by tasks. Defaults to True.
            split_retain (bool, optional): Whether to split the retain dataset by tasks. Defaults to False.

        Returns:
            DatasetDict: A dictionary containing the loaded datasets.
        """
        
        # Load data
        retain_df = pd.read_json(f'{self.data_path}/{split}/retain.jsonl', lines=True)
        forget_df = pd.read_json(f'{self.data_path}/{split}/forget.jsonl', lines=True)

        dataset = DatasetDict()

        if task == "all":
            if not split_tasks:
                dataset['retain'] = Dataset.from_pandas(retain_df)
                dataset['forget'] = Dataset.from_pandas(forget_df)
            else:
                tasks = retain_df['task'].unique()
                tasks.sort()
    
                if split_retain:
                    for task in tasks:
                        task_retain_df = retain_df[retain_df['task'] == task]
                        dataset[f'retain_{task[-1]}'] = Dataset.from_pandas(task_retain_df, preserve_index=False)
                else:
                    dataset['retain'] = Dataset.from_pandas(retain_df)
    
                for task in tasks:
                    task_forget_df = forget_df[forget_df['task'] == task]
                    dataset[f'forget_{task[-1]}'] = Dataset.from_pandas(task_forget_df, preserve_index=False)
        else:
            dataset[f'retain_{task[-1]}'] = Dataset.from_pandas(retain_df[retain_df['task'] == task], preserve_index=False)
            dataset[f'forget_{task[-1]}'] = Dataset.from_pandas(forget_df[forget_df['task'] == task], preserve_index=False)

        if self.num_samples is not None:
            dataset = self.sample_dataset(dataset)

        return dataset
        

    def sample_dataset(self, dataset_dict):
        """
        Samples a fixed number of examples from each task in the dataset.

        Args:
            dataset_dict (DatasetDict): Dictionary containing datasets.

        Returns:
            DatasetDict: A dictionary with sampled datasets.
        """
        sampled_dataset_dict = DatasetDict()
        
        for key in dataset_dict.keys():
            # Ensure that sampling is per task, respecting self.num_samples
            dataset = dataset_dict[key]
            if "task" in dataset.column_names:
                tasks = dataset.unique('task')
                sampled_rows = []
                for task in tasks:
                    task_rows = dataset.filter(lambda x: x["task"] == task)
                    sampled_rows.append(task_rows.select(
                        range(self.start_idx, self.start_idx + self.num_samples)
                    ))
                sampled_dataset_dict[key] = concatenate_datasets(sampled_rows).shuffle()
            else:
                sampled_dataset_dict[key] = dataset  # Fallback to full dataset
        
        return sampled_dataset_dict

    def tokenize_and_mask(self, dataset_dict):
        def tokenize_and_mask_function(examples):
            if self.gold_standard:
                combined_texts = [f"{p} {c}{self.tokenizer.eos_token}" for p, c in zip(examples["input"], examples["output"])]
            else:
                combined_texts = [f"{p} {c}" for p, c in zip(examples["input"], examples["output"])]

            tokenized = self.tokenizer(combined_texts, return_tensors=None)

            labels = [ids.copy() for ids in tokenized["input_ids"]]
            for i, prompt in enumerate(examples["input"]):
                prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
                labels[i][:len(prompt_ids)] = [-100] * len(prompt_ids)

            tokenized["labels"] = labels
            return tokenized

        return dataset_dict.map(
            tokenize_and_mask_function,
            batched=True,
            batch_size=None,
            #remove_columns=["id", "input", "output", "task", "split"]
        )

    def __call__(self, split, task='all', split_tasks=True, split_retain=False):
        """
        Combines all steps: creating, sampling, tokenizing, and masking the datasets.

        Args:
            split (str): Split name (e.g., "train", "test").
            split_tasks (bool, optional): Whether to split datasets by tasks. Defaults to True.
            split_retain (bool, optional): Whether to split the retain dataset by tasks. Defaults to False.

        Returns:
            DatasetDict: A ready-to-use DatasetDict for training.
        """
        assert task in ["all", "Task1", "Task2", "Task3"]
        
        dataset_dict = self.create_datasets(split, task, split_tasks, split_retain)
        return self.tokenize_and_mask(dataset_dict)
