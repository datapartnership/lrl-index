from datasets import load_dataset

# Load a specific configuration and split
dataset = load_dataset("facebook/omnilingual-asr-corpus", name="aae_Latn", split="train",)

# Get the exact number of rows
print(dataset.num_rows)
approx_rows = dataset.info.splits["train"].num_examples
print(approx_rows)
# Alternatively, check shape (num_cols, num_rows)
print(dataset.shape)
