from setuptools import setup, find_packages

setup(
    name="misalignment_detector",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "opencv-python>=4.5.0",
        "numpy>=1.19.0",
        "faster-whisper>=0.9.0",
        "ollama>=0.1.0",
        "websockets>=10.4",
        "matplotlib>=3.5.0",
        "pillow>=9.0.0",
        "jsonschema>=4.0.0",
        "pandas>=1.3.0",
        "sounddevice>=0.4.0",
        "soundfile>=0.10.0",
        "scipy>=1.7.0"
    ],
    author="AI Engineer",
    author_email="user@example.com",
    description="A system to detect misalignment and confusion in real-time conversations",
    keywords="misalignment, conversation, analysis, AI",
    python_requires=">=3.9",
)#!/usr/bin/env python3
"""
Setup script for the misalignment detection system.
"""
from setuptools import setup, find_packages
import os

# Read requirements
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# Read README for long description
if os.path.exists('README.md'):
    with open('README.md', encoding='utf-8') as f:
        long_description = f.read()
else:
    long_description = "Misalignment Detection System"

setup(
    name="misalignment_detector",
    version="1.0.0",
    description="Real-time system for detecting communication misalignment",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="AI Misalignment Detection Team",
    author_email="example@example.com",
    url="https://github.com/example/misalignment-detector",
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'misalignment-detector=main:main',
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
)