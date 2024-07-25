# -*- coding: utf-8 -*-
# Copyright (c) Louis Brulé Naudet. All Rights Reserved.
# This software may be used and distributed according to the terms of License Agreement.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
import uuid

from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Type,
    Tuple,
    Union,
    Mapping,
    TypeVar,
    Callable,
    Optional,
    Sequence,
)

import datasets
import numpy as np

from hf_for_legal._decorators import (
    memory, 
    timer
)

class DatasetFormatter:
    """
    A class used to format datasets by adding hash and UUID columns, as well as additional utility functions.

    This class provides methods to add a SHA-256 hash column and a UUID column
    to a Hugging Face `datasets.Dataset` object, normalize text, filter rows,
    rename columns, drop columns, add constant value columns, convert column types,
    handle missing values, and compute summary statistics.

    Parameters
    ----------
    dataset : datasets.Dataset
        The dataset to be formatted.

    Methods
    -------
    hash(column_name: str = "document", hash_column_name: str = "hash") -> datasets.Dataset
        Creates a SHA-256 hash column for the dataset.
    
    uuid(uuid_column_name: str = "uuid") -> datasets.Dataset
        Adds a UUID column to the dataset.
    
    normalize_text(column_name: str, normalized_column_name: Optional[str] = None) -> datasets.Dataset
        Normalizes text in a specified column.
    
    filter_rows(condition: Callable) -> datasets.Dataset
        Filters rows based on a given condition.
    
    rename_column(old_column_name: str, new_column_name: str) -> datasets.Dataset
        Renames a column in the dataset.
    
    drop_column(column_name: str) -> datasets.Dataset
        Drops a specified column from the dataset.
    
    add_constant_column(column_name: str, constant_value) -> datasets.Dataset
        Adds a new column with a constant value.
    
    convert_column_type(column_name: str, new_type: Union[type, str]) -> datasets.Dataset
        Converts a column to a specified data type.
    
    fill_missing(column_name: str, fill_value) -> datasets.Dataset
        Fills missing values in a column with a specified value.
    
    compute_summary(column_name: str) -> Dict[str, float]
        Computes summary statistics for a numerical column.
    
    __call__(hash_column_name: str = "hash", uuid_column_name: str = "uuid") -> datasets.Dataset
        Applies both the hash and UUID functions to the dataset.
    """
    def __init__(
        self, 
        dataset: datasets.Dataset
    ):
        """
        Initializes the DatasetFormatter with a dataset.

        Parameters
        ----------
        dataset : datasets.Dataset
            The dataset to be formatted.
        """
        if not isinstance(dataset, datasets.Dataset):
            raise TypeError("Expected a Hugging Face `datasets.Dataset` object.")

        self.dataset = dataset


    @memory(print_report=True)
    @timer(print_time=True)
    def hash(
        self, 
        column_name: str = "document", 
        hash_column_name: str = "hash"
    ) -> datasets.Dataset:
        """
        Adds a SHA-256 hash column to the dataset.

        This function takes the document content from the specified column in the dataset,
        converts it to a string, encodes it in UTF-8, and then generates a SHA-256 hash of
        the encoded text.

        Parameters
        ----------
        column_name : str, optional
            The name of the column containing the text to be hashed. Default is "document".

        hash_column_name : str, optional
            The name of the new column to store the hash values. Default is "hash".

        Returns
        -------
        datasets.Dataset
            The dataset with the added hash column.
        """
        def generate_hash(
            text: str
        ) -> str:
            """
            Generate a SHA-256 hash for a given text.

            Parameters
            ----------
            text : str
                The text content to be hashed.

            Returns
            -------
            str
                The SHA-256 hash of the input text, represented as a hexadecimal string.
            """
            return hashlib.sha256(str(text).encode()).hexdigest()


        if column_name not in self.dataset.column_names:
            raise ValueError(f"Column '{column_name}' does not exist in the dataset.")

        # Apply the hash generation to each row in the dataset
        return self.dataset.map(lambda x: {hash_column_name: generate_hash(x[column_name])})


    @memory(print_report=True)
    @timer(print_time=True)
    def uuid(
        self, 
        uuid_column_name: str = "uuid"
    ) -> datasets.Dataset:
        """
        Adds a UUID column to the dataset.

        This function generates a UUID for each entry in the dataset and adds it
        as a new column named "uuid".

        Parameters
        ----------
        uuid_column_name : str, optional
            The name of the new column to store the UUID values. Default is "uuid".

        Returns
        -------
        datasets.Dataset
            The dataset with the added UUID column.
        """
        # Apply the UUID generation to each row in the dataset
        return self.dataset.map(lambda x: {uuid_column_name: str(uuid.uuid4())})


    @memory(print_report=True)
    @timer(print_time=True)
    def normalize_text(
        self, 
        column_name: str, 
        normalized_column_name: Optional[str] = None
    ) -> datasets.Dataset:
        """
        Normalizes text in a specified column by converting to lowercase and stripping whitespace.

        Parameters
        ----------
        column_name : str
            The name of the column containing the text to be normalized.
        
        normalized_column_name : str, optional
            The name of the new column to store the normalized text. If not provided, it overwrites the original column.

        Returns
        -------
        datasets.Dataset
            The dataset with the normalized text column.
        """
        def normalize(
            text: str
        ) -> str:
            """
            Normalize the text by converting to lowercase and stripping leading/trailing whitespace.

            Parameters
            ----------
            text : str
                The text content to be normalized.

            Returns
            -------
            str
                The normalized text.
            """
            return text.lower().strip()


        if column_name not in self.dataset.column_names:
            raise ValueError(f"Column '{column_name}' does not exist in the dataset.")

        new_column_name = normalized_column_name if normalized_column_name else column_name

        # Apply text normalization to each row in the dataset
        return self.dataset.map(lambda x: {new_column_name: normalize(x[column_name])})


    @memory(print_report=True)
    @timer(print_time=True)
    def filter_rows(
        self, 
        condition: Callable
    ) -> datasets.Dataset:
        """
        Filters rows based on a given condition.

        Parameters
        ----------
        condition : Callable
            A function that takes a row (dict) and returns True if the row should be included in the filtered dataset.

        Returns
        -------
        datasets.Dataset
            The filtered dataset.
        """
        # Apply row filtering based on the condition
        return self.dataset.filter(condition)


    def rename_column(
        self, 
        old_column_name: str, 
        new_column_name: str
    ) -> datasets.Dataset:
        """
        Renames a column in the dataset.

        Parameters
        ----------
        old_column_name : str
            The current name of the column to be renamed.
        
        new_column_name : str
            The new name for the column.

        Returns
        -------
        datasets.Dataset
            The dataset with the renamed column.
        """
        if old_column_name not in self.dataset.column_names:
            raise ValueError(f"Column '{old_column_name}' does not exist in the dataset.")

        # Rename the column
        return self.dataset.rename_column(old_column_name, new_column_name)


    def drop_column(
        self, 
        column_name: str
    ) -> datasets.Dataset:
        """
        Drops a specified column from the dataset.

        Parameters
        ----------
        column_name : str
            The name of the column to be dropped.

        Returns
        -------
        datasets.Dataset
            The dataset with the specified column dropped.
        """
        if column_name not in self.dataset.column_names:
            raise ValueError(f"Column '{column_name}' does not exist in the dataset.")

        # Drop the column
        return self.dataset.remove_columns([column_name])


    def add_constant_column(
        self, 
        column_name: str, 
        constant_value
    ) -> datasets.Dataset:
        """
        Adds a new column with a constant value.

        Parameters
        ----------
        column_name : str
            The name of the new column to be added.
        
        constant_value
            The constant value to be assigned to each row in the new column.

        Returns
        -------
        datasets.Dataset
            The dataset with the new constant value column.
        """
        # Add the constant value column
        return self.dataset.map(lambda x: {column_name: constant_value})


    @memory(print_report=True)
    @timer(print_time=True)
    def convert_column_type(
        self, 
        column_name: str, 
        new_type: Union[type, str]
    ) -> datasets.Dataset:
        """
        Converts a column to a specified data type.

        Parameters
        ----------
        column_name : str
            The name of the column to be converted.
        
        new_type : Union[type, str]
            The new data type for the column, e.g., int, float, str.

        Returns
        -------
        datasets.Dataset
            The dataset with the converted column.
        """
        if column_name not in self.dataset.column_names:
            raise ValueError(f"Column '{column_name}' does not exist in the dataset.")

        # Apply the type conversion to each row in the dataset
        return self.dataset.map(lambda x: {column_name: new_type(x[column_name])})


    @memory(print_report=True)
    @timer(print_time=True)
    def fill_missing(
        self, 
        column_name: str, 
        fill_value
    ) -> datasets.Dataset:
        """
        Fills missing values in a column with a specified value.

        Parameters
        ----------
        column_name : str
            The name of the column with missing values to be filled.

        fill_value
            The value to fill in for missing values.

        Returns
        -------
        datasets.Dataset
            The dataset with missing values filled.
        """
        if column_name not in self.dataset.column_names:
            raise ValueError(f"Column '{column_name}' does not exist in the dataset.")

        # Fill missing values in the column
        return self.dataset.map(lambda x: {column_name: x[column_name] if x[column_name] is not None else fill_value})


    @memory(print_report=True)
    @timer(print_time=True)
    def compute_summary(
        self, 
        column_name: str
    ) -> Dict[str, float]:
        """
        Computes summary statistics for a numerical column.

        Parameters
        ----------
        column_name : str
            The name of the numerical column to compute summary statistics for.

        Returns
        -------
        Dict[str, float]
            A dictionary containing summary statistics (mean, median, std) for the column.
        """
        if column_name not in self.dataset.column_names:
            raise ValueError(f"Column '{column_name}' does not exist in the dataset.")

        column_data = np.array(self.dataset[column_name])

        summary = {
            "mean": np.mean(column_data),
            "median": np.median(column_data),
            "std": np.std(column_data)
        }

        return summary


    @memory(print_report=True)
    @timer(print_time=True)
    def __call__(
        self, 
        hash_column_name: str = "hash", 
        uuid_column_name: str = "uuid"
    ) -> datasets.Dataset:
        """
        Applies both the hash and UUID functions to the dataset.

        This method first adds a SHA-256 hash column and then adds a UUID column to the dataset.

        Parameters
        ----------
        hash_column_name : str, optional
            The name of the new column to store the hash values. Default is "hash".
        
        uuid_column_name : str, optional
            The name of the new column to store the UUID values. Default is "uuid".

        Returns
        -------
        datasets.Dataset
            The dataset with both hash and UUID columns.
        """
        # Add the hash column
        self.dataset = self.hash(hash_column_name=hash_column_name)
        # Add the UUID column
        self.dataset = self.uuid(uuid_column_name=uuid_column_name)
        
        return self.dataset