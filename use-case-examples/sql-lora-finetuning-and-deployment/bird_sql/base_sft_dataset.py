from abc import ABC, abstractmethod
from transformers import AutoTokenizer
from datasets import Dataset


class BaseSFTDataset(ABC):

    def __init__(
        self,
        model_id_to_prep_for: str,
        max_seq_len: int,  # The maximum allowed length of each sample (input tokens + output tokens)
        num_workers: int = 20,
        seed: int = 186,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model_to_prep_for = model_id_to_prep_for

        self.tokenizer = AutoTokenizer.from_pretrained(model_id_to_prep_for)
        self.max_seq_len = max_seq_len
        self.num_workers = num_workers
        self.seed = seed
        self.eot_marker = self._determine_eot_marker()

    def make_dataset(self) -> Dataset:
        """
        Create a processed dataset.
        """
        dataset = self._load_dataset()
        # Save the name of columns (will remove these later)
        cols_to_remove = dataset.column_names
        # Prepare
        dataset = self._prepare_dataset(dataset)

        # Exclude columns to remove
        if "input" in cols_to_remove:
            cols_to_remove.remove("input")
        if "output" in cols_to_remove:
            cols_to_remove.remove("output")

        # Add a "text" column that is input + output for easier processing later
        def _concat_input_output(row):
            return {"text": row["input"] + row["output"]}
        dataset = dataset.map(
            _concat_input_output,
            num_proc=self.num_workers,
        )

        # Add length column
        def _compute_length(row):
            tokens = self.tokenizer(row["text"]).input_ids
            return {"length": len(tokens)}
        dataset = dataset.map(
            _compute_length,
            num_proc=self.num_workers,
        )

        # Filter based on token counts (only keep those that fit within max_seq_len)
        dataset = dataset.filter(
            lambda row: row["length"] <= self.max_seq_len,
            num_proc=self.num_workers,
        )

        # Remove irrelevant columns
        dataset = dataset.remove_columns(cols_to_remove)

        return dataset

    def _determine_eot_marker(self):
        """
        Figures out the end-of-turn marker of the tokenizer. This is helpful for unifying tokenizers
        that add extra stuff besides EOS in the end when applying chat templates.
        """
        distinct_marker = "MY_MARKER_TO_SEARCH_FOR"
        template_applied = self.tokenizer.apply_chat_template(
            [{"role": "assistant", "content": distinct_marker}],
            tokenize=False,
        )
        return template_applied.split(distinct_marker)[-1]

    def _get_prompt_with_chat_template_applied(
        self,
        system_message: str,
        user_message: str,
        enable_reasoning: bool,
        add_generation_prompt: bool = True,  # Will be forwarded to tokenizer.apply_chat_template
    ):
        if "NVIDIA-Nemotron-3" in self.model_to_prep_for:
            # Nemotron Nano 3 style
            prompt = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_message.strip()},
                    {"role": "user", "content": user_message.strip()},
                ],
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=enable_reasoning,
            )
        else:
            raise NotImplementedError(f"Dataprep for the model {self.model_to_prep_for} is not currently supported.")

        return prompt

    @abstractmethod
    def _load_dataset(self) -> Dataset:
        """
        Load the HFDataset from source.
        """
        pass

    @abstractmethod
    def _prepare_dataset(self, dataset: Dataset):
        """
        Apply templates, etc to form the input/output dataset
        """
        pass
