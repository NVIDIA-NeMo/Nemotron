from datasets import Dataset, load_dataset
from base_sft_dataset import BaseSFTDataset

USER_MESSAGE_TEMPLATE = """
{schema}

{question}
{evidence}
"""

class DatasetBIRDReasoning(BaseSFTDataset):

    def _apply_chat_template(self, row, idx):

        schema = row["schema"]
        question = row["question"]
        evidence = row["evidence"]
        sql = row["SQL"]
        reasoning = row["reasoning_trace"].strip()

        user_message = USER_MESSAGE_TEMPLATE.format(
            schema=schema,
            question=question,
            evidence=evidence,
        ).strip()
        # Match Ultra's canonical thinking turn exactly: <think>\n{reasoning}</think>{sql}
        # (the prompt already ends with an open "<think>\n"). No extra whitespace around </think>.
        assistant_response = f"{reasoning}</think>{sql}".strip()
        prompt = self._get_prompt_with_chat_template_applied(
            system_message="",
            user_message=user_message,
            enable_reasoning=True,
        )
        completion = assistant_response + self.eot_marker
        return {
            "input": prompt,
            "output": completion,
        }

    def _load_dataset(self) -> Dataset:
        return load_dataset("meowterspace45/bird-sql-train-with-reasoning", split="train")

    def _prepare_dataset(self, dataset: Dataset):
        dataset = dataset.map(
            self._apply_chat_template,
            with_indices=True,
            load_from_cache_file=False,
            num_proc=self.num_workers,
        )

        return dataset
