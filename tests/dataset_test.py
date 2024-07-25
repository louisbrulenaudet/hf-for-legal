# -*- coding: utf-8 -*-
# Copyright (c) Louis Brulé Naudet. All Rights Reserved.
# This software may be used and distributed according to the terms of License Agreement.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

import hashlib
import uuid
import datasets
import pytest

from datasets import Dataset

from src.hf_for_legal import DatasetFormatter


@pytest.fixture
def sample_dataset():
    """
    Fixture for creating a sample dataset.

    Returns
    -------
    datasets.Dataset
        A sample dataset containing documents for testing.
    """
    data = {
        "document": [
            "This is a test document.", 
            "Another test document."
        ]
    }
    dataset = datasets.Dataset.from_dict(data)

    return dataset


def test_hash(
    sample_dataset: Dataset
):
    """
    Test the `hash` method of `DatasetFormatter`.

    Parameters
    ----------
    sample_dataset : datasets.Dataset
        The sample dataset fixture.

    Asserts
    -------
    Asserts that the hash column is correctly added with SHA-256 hash values.
    """

    formatter = DatasetFormatter(sample_dataset)

    formatted_dataset = formatter.hash(
        column_name="document", 
        hash_column_name="hash"
    )

    assert "hash" in formatted_dataset.column_names

    expected_hash = hashlib.sha256(
        "This is a test document.".encode()
    ).hexdigest()

    assert formatted_dataset[0]["hash"] == expected_hash


def test_uuid(
    sample_dataset: Dataset
):
    """
    Test the `uuid` method of `DatasetFormatter`.

    Parameters
    ----------
    sample_dataset : datasets.Dataset
        The sample dataset fixture.

    Asserts
    -------
    Asserts that the UUID column is correctly added with UUID values.
    """
    formatter = DatasetFormatter(sample_dataset)
    formatted_dataset = formatter.uuid(
        uuid_column_name="uuid"
    )

    assert "uuid" in formatted_dataset.column_names
    assert len(formatted_dataset[0]["uuid"]) == 36  # UUID length


def test_normalize_text(
    sample_dataset: Dataset
):
    """
    Test the `normalize_text` method of `DatasetFormatter`.

    Parameters
    ----------
    sample_dataset : datasets.Dataset
        The sample dataset fixture.

    Asserts
    -------
    Asserts that the text in the specified column is normalized.
    """
    formatter = DatasetFormatter(sample_dataset)
    formatted_dataset = formatter.normalize_text(
        column_name="document", 
        normalized_column_name="normalized_text"
    )

    assert "normalized_text" in formatted_dataset.column_names
    assert formatted_dataset[0]["normalized_text"] == "this is a test document."


def test_filter_rows(
    sample_dataset: Dataset
):
    """
    Test the `filter_rows` method of `DatasetFormatter`.

    Parameters
    ----------
    sample_dataset : datasets.Dataset
        The sample dataset fixture.

    Asserts
    -------
    Asserts that rows are correctly filtered based on the condition.
    """
    formatter = DatasetFormatter(sample_dataset)
    formatted_dataset = formatter.filter_rows(
        lambda x: "Another" in x["document"]
    )

    assert len(formatted_dataset) == 1
    assert formatted_dataset[0]["document"] == "Another test document."


def test_rename_column(
    sample_dataset: Dataset
):
    """
    Test the `rename_column` method of `DatasetFormatter`.

    Parameters
    ----------
    sample_dataset : datasets.Dataset
        The sample dataset fixture.

    Asserts
    -------
    Asserts that the column is correctly renamed.
    """
    formatter = DatasetFormatter(sample_dataset)
    formatted_dataset = formatter.rename_column(
        old_column_name="document",
        new_column_name="doc"
    )

    assert "doc" in formatted_dataset.column_names
    assert "document" not in formatted_dataset.column_names


def test_drop_column(
    sample_dataset: Dataset
):
    """
    Test the `drop_column` method of `DatasetFormatter`.

    Parameters
    ----------
    sample_dataset : datasets.Dataset
        The sample dataset fixture.

    Asserts
    -------
    Asserts that the specified column is correctly dropped.
    """
    formatter = DatasetFormatter(sample_dataset)
    formatted_dataset = formatter.drop_column(
        column_name="document"
    )

    assert "document" not in formatted_dataset.column_names


def test_add_constant_column(
    sample_dataset: Dataset
):
    """
    Test the `add_constant_column` method of `DatasetFormatter`.

    Parameters
    ----------
    sample_dataset : datasets.Dataset
        The sample dataset fixture.

    Asserts
    -------
    Asserts that the new constant value column is correctly added.
    """
    formatter = DatasetFormatter(sample_dataset)
    formatted_dataset = formatter.add_constant_column(
        column_name="new_column", 
        constant_value="constant_value"
    )

    assert "new_column" in formatted_dataset.column_names
    assert formatted_dataset[0]["new_column"] == "constant_value"


def test_convert_column_type(
    sample_dataset: Dataset
):
    """
    Test the `convert_column_type` method of `DatasetFormatter`.

    Parameters
    ----------
    sample_dataset : datasets.Dataset
        The sample dataset fixture with an added column for testing.

    Asserts
    -------
    Asserts that the column type is correctly converted.
    """
    sample_dataset = sample_dataset.add_column(
        "number_str", 
        [
            "1", 
            "2"
        ]
    )
    formatter = DatasetFormatter(sample_dataset)
    formatted_dataset = formatter.convert_column_type(
        column_name="number_str", 
        new_type=int
    )

    assert formatted_dataset[0]["number_str"] == 1
    assert isinstance(formatted_dataset[0]["number_str"], int)


def test_fill_missing():
    """
    Test the `fill_missing` method of `DatasetFormatter`.

    Asserts
    -------
    Asserts that missing values in the specified column are correctly filled.
    """
    data = {
        "some_column": [1, None, 3]
    }
    dataset = datasets.Dataset.from_dict(data)
    formatter = DatasetFormatter(dataset)
    formatted_dataset = formatter.fill_missing(
        column_name="some_column", 
        fill_value=0
    )

    assert formatted_dataset[1]["some_column"] == 0

def test_compute_summary():
    """
    Test the `compute_summary` method of `DatasetFormatter`.

    Asserts
    -------
    Asserts that the summary statistics for a numerical column are correctly computed.
    """
    data = {
        "numerical_column": [1, 2, 3, 4, 5]
    }
    dataset = datasets.Dataset.from_dict(data)
    formatter = DatasetFormatter(dataset)
    
    summary_stats = formatter.compute_summary(
        column_name="numerical_column"
    )

    assert summary_stats["mean"] == 3.0
    assert summary_stats["median"] == 3.0
    assert summary_stats["std"] == pytest.approx(1.414, 0.001)


def test_call(
    sample_dataset: Dataset
):
    """
    Test the `__call__` method of `DatasetFormatter`.

    Parameters
    ----------
    sample_dataset : datasets.Dataset
        The sample dataset fixture.

    Asserts
    -------
    Asserts that both hash and UUID columns are correctly added.
    """
    formatter = DatasetFormatter(sample_dataset)
    formatted_dataset = formatter(
        hash_column_name="hash", 
        uuid_column_name="uuid"
    )

    assert "hash" in formatted_dataset.column_names
    assert "uuid" in formatted_dataset.column_names

    expected_hash = hashlib.sha256(
        "This is a test document.".encode()
    ).hexdigest()

    assert formatted_dataset[0]["hash"] == expected_hash
    assert len(formatted_dataset[0]["uuid"]) == 36  # UUID length
