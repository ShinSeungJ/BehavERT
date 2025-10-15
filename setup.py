"""
Setup script for BehavERT package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="behavert",
    version="1.0.0",
    author="BehavERT Team",
    author_email="contact@behavert.org",
    description="BERT-based Animal Behavior Analysis from Keypoint Data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/BehavERT",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "pre-commit>=2.0",
        ],
        "notebooks": [
            "jupyter>=1.0",
            "matplotlib>=3.3",
            "seaborn>=0.11",
        ],
    },
    entry_points={
        "console_scripts": [
            "behavert-pretrain=scripts.pretrain:main",
            "behavert-finetune=scripts.finetune:main",
            "behavert-evaluate=scripts.evaluate:main",
            "behavert-inference=scripts.inference:main",
        ],
    },
)
