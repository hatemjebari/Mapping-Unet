from setuptools import setup, find_packages

setup(
    name="data_prep",
    version="0.1.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    python_requires=">=3.6",
    install_requires=[
        "pandas",
        "dill",
        "fastprogress",
        "matplotlib",
        "neptune-client",
        "optuna",
        "transformers",
        "tables",
        "numpy"
    ],
)
