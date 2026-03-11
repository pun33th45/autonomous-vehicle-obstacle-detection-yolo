"""
setup.py — makes the project installable as a Python package.

Install in editable (dev) mode:
    pip install -e .
"""

from setuptools import find_packages, setup

setup(
    name="autonomous-obstacle-detection",
    version="1.0.0",
    description="YOLOv8-based obstacle detection for autonomous vehicles",
    author="Your Name",
    python_requires=">=3.10",
    packages=find_packages(exclude=["tests*", "notebooks*"]),
    install_requires=[
        "torch>=2.1.0",
        "ultralytics>=8.0.200",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "PyYAML>=6.0.1",
        "tqdm>=4.66.0",
        "matplotlib>=3.7.0",
    ],
    extras_require={
        "api": ["fastapi>=0.104.0", "uvicorn[standard]>=0.24.0", "python-multipart>=0.0.6"],
        "dev": ["pytest>=7.4.0", "black>=23.9.0", "isort>=5.12.0"],
        "hpo": ["optuna>=3.4.0"],
        "onnx": ["onnx>=1.14.0", "onnxruntime>=1.16.0"],
    },
)
