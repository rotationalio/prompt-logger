from setuptools import setup, find_packages
from prompt_logger.version import get_version

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="prompt-logger",
    version=get_version(),
    author="Patrick Deziel",
    author_email="deziel.patrick@gmail.com",
    description="A tool for logging and exporting AI prompts and responses",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rotationalio/prompt-logger",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "sqlalchemy>=2.0.0",
    ],
    entry_points={
        "console_scripts": [
            "prompt-logger=prompt_logger.cli:main",
        ],
    },
)
