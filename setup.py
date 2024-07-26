# -*- coding: utf-8 -*-
# Copyright (c) Louis Brulé Naudet. All Rights Reserved.
# This software may be used and distributed according to the terms of License Agreement.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import (
    setup, 
    find_packages
)

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

setup(
    name="hf-for-legal",
    version="0.0.12",
    description="HF for Legal, a community package for legal application 🤗",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Louis Brulé Naudet",
    author_email="louisbrulenaudet@icloud.com",
    url="https://github.com/louisbrulenaudet/hf-for-legal",
    project_urls={
        "Homepage": "https://github.com/louisbrulenaudet/hf-for-legal",
        "Repository": "https://github.com/louisbrulenaudet/hf-for-legal",
    },
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords=[
        "language-models", "retrieval", "web-scraping", "gpl", "nlp",
        "hf-for-legal", "machine-learning", "retrieval-augmented-generation",
        "RAG", "huggingface", "generative-ai", "llama", "Mistral",
        "inference-api", "datasets", "llm-as-judge"
    ],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "datasets",
        "numpy",
        "tqdm"
    ],
)
