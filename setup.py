import re
from pathlib import Path

from setuptools import find_namespace_packages, setup

_here = Path(__file__).resolve().parent

name = "jaxrl"

# Reference: https://github.com/patrick-kidger/equinox/blob/main/setup.py
with open(_here / name / "__init__.py") as f:
    meta_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    if meta_match:
        version = meta_match.group(1)
    else:
        raise RuntimeError("Unable to find __version__ string.")


with open(_here / "README.md", "r") as f:
    readme = f.read()

core_requirements = [
    "flax",
    "optax",
    "distrax",
    "typing_extensions",
    "tensorflow",
    "protobuf==3.20.0",
]

testing_requirements = [
    "absl-py",
    "pytest",
]

dev_requirements = [
    "black",
    "isort",
    "flake8",
    "mypy",
    "ruff",
] + testing_requirements

classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Natural Language :: English",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3 :: Only",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]

author = "Kevin Zakka"

author_email = "kevinarmandzakka@gmail.com"

description = "A tiny reinforcement learning library written in JaX"


setup(
    name=name,
    version=version,
    author=author,
    author_email=author_email,
    maintainer=author,
    maintainer_email=author_email,
    description=description,
    long_description=readme,
    long_description_content_type="text/markdown",
    url=f"https://github.com/kevinzakka/{name}",
    license="Apache License 2.0",
    license_files=("LICENSE",),
    packages=find_namespace_packages(exclude=["*_test.py"]),
    package_data={
        f"{name}": [
            "py.typed",
        ]
    },
    zip_safe=False,
    python_requires=">=3.10",
    install_requires=core_requirements,
    classifiers=classifiers,
    extras_require={
        "testing": testing_requirements,
        "dev": dev_requirements,
    },
)
