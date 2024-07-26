<img src="https://huggingface.co/spaces/HFforLegal/README/resolve/main/assets/thumbnail.png">

# HF for Legal: A Community Package for Legal Applications 🤗

[![Python](https://img.shields.io/pypi/pyversions/tensorflow.svg)](https://badge.fury.io/py/tensorflow) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) ![Maintainer](https://img.shields.io/badge/maintainer-@louisbrulenaudet-blue)

Welcome to the HF for Legal package, a library dedicated to breaking down the opacity of language models for legal professionals. Our mission is to empower legal practitioners, scholars, and researchers with the knowledge and tools they need to navigate the complex world of AI in the legal domain. At HF for Legal, we aim to:
- Demystify AI language models for the legal community
- Share curated resources, including specialized legal models, datasets, and tools
- Foster collaboration on projects that enhance legal research and practice through AI
- Provide a platform for discussing ethical implications and best practices of AI in law
- Offer tutorials and workshops on leveraging AI technologies in legal work

By bringing together legal experts, AI researchers, and technology enthusiasts, we strive to create an open ecosystem where legal professionals can easily access, understand, and utilize AI models tailored to their needs. Whether you're a practicing attorney, a legal scholar, or a technologist interested in legal applications of AI, HF for Legal is your hub for exploration, learning, and innovation in the evolving landscape of AI-assisted legal practice.

## Installation

To use hf-for-legal, you need to have the following Python packages installed:
- `numpy`
- `datasets`
- `tqdm`

You can install these packages via pip:

```bash
pip install numpy datasets hf-for-legal tqdm
```

## Usage

First, initialize the DatasetFormatter class with your dataset:

```python
import datasets
from hf_for_legal import DatasetFormatter

# Load a sample dataset
dataset = datasets.Dataset.from_dict(
  {
    "document": [
      "This is a test document.", 
      "Another test document.
    ]
  }
)

# Create an instance of DatasetFormatter
formatter = DatasetFormatter(dataset)

# Apply the hash and UUID functions
formatted_dataset = formatter()
print(formatted_dataset)
```

# Class: DatasetFormatter

## Parameters:

- **dataset** (`datasets.Dataset`): The dataset to be formatted.

## Attributes:

- **dataset** (`datasets.Dataset`): The original dataset.

## Methods

### hash(self, column_name: str = "document", hash_column_name: str = "hash") -> datasets.Dataset

Add a SHA-256 hash column to the dataset.

#### Parameters:

- **column_name** (`str`, optional): The name of the column containing the text to hash. Default is "document".
- **hash_column_name** (`str`, optional): The name of the column to store the hash values. Default is "hash".

#### Returns:

- `datasets.Dataset`: The dataset with the new hash column.

#### Raises:

- **ValueError**: If the specified column_name does not exist in the dataset.

### uuid(self, uuid_column_name: str = "uuid") -> datasets.Dataset

Add a UUID column to the dataset.

#### Parameters:

- **uuid_column_name** (`str`, optional): The name of the column to store the UUID values. Default is "uuid".

#### Returns:

- `datasets.Dataset`: The dataset with the new UUID column.

### normalize_text(self, column_name: str, normalized_column_name: Optional[str] = None) -> datasets.Dataset

Normalize text in a specified column by converting to lowercase and stripping whitespace.

#### Parameters:

- **column_name** (`str`): The name of the column containing the text to be normalized.
- **normalized_column_name** (`str`, optional): The name of the new column to store the normalized text. If not provided, it overwrites the original column.

#### Returns:

- `datasets.Dataset`: The dataset with the normalized text column.

#### Raises:

- **ValueError**: If the specified column_name does not exist in the dataset.

### filter_rows(self, condition: Callable) -> datasets.Dataset

Filter rows based on a given condition.

#### Parameters:

- **condition** (`Callable`): A function that takes a row (dict) and returns True if the row should be included in the filtered dataset.

#### Returns:

- `datasets.Dataset`: The filtered dataset.

### rename_column(self, old_column_name: str, new_column_name: str) -> datasets.Dataset

Rename a column in the dataset.

#### Parameters:

- **old_column_name** (`str`): The current name of the column to be renamed.
- **new_column_name** (`str`): The new name for the column.

#### Returns:

- `datasets.Dataset`: The dataset with the renamed column.

#### Raises:

- **ValueError**: If the specified old_column_name does not exist in the dataset.

### drop_column(self, column_name: str) -> datasets.Dataset

Drop a specified column from the dataset.

#### Parameters:

- **column_name** (`str`): The name of the column to be dropped.

#### Returns:

- `datasets.Dataset`: The dataset with the specified column dropped.

#### Raises:

- **ValueError**: If the specified column_name does not exist in the dataset.

### add_constant_column(self, column_name: str, constant_value) -> datasets.Dataset

Add a new column with a constant value.

#### Parameters:

- **column_name** (`str`): The name of the new column to be added.
- **constant_value**: The constant value to be assigned to each row in the new column.

#### Returns:

- `datasets.Dataset`: The dataset with the new constant value column.

### convert_column_type(self, column_name: str, new_type: Union[type, str]) -> datasets.Dataset

Convert a column to a specified data type.

#### Parameters:

- **column_name** (`str`): The name of the column to be converted.
- **new_type** (`Union[type, str]`): The new data type for the column, e.g., int, float, str.

#### Returns:

- `datasets.Dataset`: The dataset with the converted column.

#### Raises:

- **ValueError**: If the specified column_name does not exist in the dataset.

### fill_missing(self, column_name: str, fill_value) -> datasets.Dataset

Fill missing values in a column with a specified value.

#### Parameters:

- **column_name** (`str`): The name of the column with missing values to be filled.
- **fill_value**: The value to fill in for missing values.

#### Returns:

- `datasets.Dataset`: The dataset with missing values filled.

#### Raises:

- **ValueError**: If the specified column_name does not exist in the dataset.

### compute_summary(self, column_name: str) -> Dict[str, float]

Compute summary statistics for a numerical column.

#### Parameters:

- **column_name** (`str`): The name of the numerical column to compute summary statistics for.

#### Returns:

- **Dict[str, float]**: A dictionary containing summary statistics (mean, median, std) for the column.

#### Raises:

- **ValueError**: If the specified column_name does not exist in the dataset.

### __call__(self, hash_column_name: str = "hash", uuid_column_name: str = "uuid") -> datasets.Dataset

Apply both the hash and UUID functions to the dataset.

#### Parameters:

- **hash_column_name** (`str`, optional): The name of the new column to store the hash values. Default is "hash".
- **uuid_column_name** (`str`, optional): The name of the new column to store the UUID values. Default is "uuid".

#### Returns:

- `datasets.Dataset`: The dataset with both hash and UUID columns.

## Community Discord

You can now join, communicate and share on the HF for Legal community server on Discord.

Link to the server: https://discord.gg/vNhXRsfw

This server will simplify communication between members of the organization and generate synergies around the various projects in the three areas of interactive applications, databases and models.

An example of a project soon to be published: a duplication of the Laws database, but this time containing embeddings already calculated for different models, to enable simplified integration within Spaces (RAG chatbot ?) and save deployment costs for users wishing to use these technologies for their professional and personal projects.

## Citing & Authors

If you use this code in your research, please use the following BibTeX entry.

```BibTeX
@misc{louisbrulenaudet2024,
  author =       {Louis Brulé Naudet},
  title =        {HF for Legal: A Community Package for Legal Applications},
  year =         {2024}
  howpublished = {\url{https://github.com/louisbrulenaudet/hf-for-legal}},
}
```

## Feedback

If you have any feedback, please reach out at [louisbrulenaudet@icloud.com](mailto:louisbrulenaudet@icloud.com).