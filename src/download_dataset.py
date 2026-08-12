from huggingface_hub import hf_hub_download
from datasets import load_dataset

REPO_ID = "openslr/librispeech_asr"

# --------------------------------
# 1. Download 5 training shards
# --------------------------------

train_files = []

for i in range(5):
    filename = f"clean/train.100/{i:04d}.parquet"

    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type="dataset",
        local_dir="data/librispeech"
    )

    train_files.append(path)


# --------------------------------
# 2. Download validation
# --------------------------------

validation_path = hf_hub_download(
    repo_id=REPO_ID,
    filename="clean/validation/0000.parquet",
    repo_type="dataset",
    local_dir="data/librispeech"
)


# --------------------------------
# 3. Download test
# --------------------------------

test_path = hf_hub_download(
    repo_id=REPO_ID,
    filename="clean/test/0000.parquet",
    repo_type="dataset",
    local_dir="data/librispeech"
)


# --------------------------------
# 4. Load them as datasets
# --------------------------------

train = load_dataset(
    "parquet",
    data_files=train_files,
    split="train"
)

validation = load_dataset(
    "parquet",
    data_files=validation_path,
    split="train"
)

test = load_dataset(
    "parquet",
    data_files=test_path,
    split="train"
)


# --------------------------------
# 5. Print information
# --------------------------------

print("Training examples:", len(train))
print("Validation examples:", len(validation))
print("Test examples:", len(test))

print("\nExample:")
print(train[0])
