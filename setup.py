#!/usr/bin/env python3
"""
Setup script for Semantic Segmentation Benchmark
"""
from setuptools import setup, find_packages

setup(
    name="semantic-segmentation-benchmark",
    version="1.0.0",
    description="Benchmark for semantic segmentation models on satellite data",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "torch>=1.12.0",
        "torchvision>=0.13.0",
        "ultralytics>=8.0.0",
        "numpy>=1.21.0",
        "Pillow>=9.0.0",
        "opencv-python>=4.5.0",
        "scikit-image>=0.19.0",
        "scikit-learn>=1.0.0",
        "joblib>=1.2.0",
        "PyYAML>=6.0",
        "tqdm>=4.64.0",
        "matplotlib>=3.5.0",
        "pandas>=1.3.0",
        "reportlab",
    ],
    python_requires=">=3.8",
)
