# Project Setup and Installation Guide

## Prerequisites
- Python 3.9 - 3.11
- pip (Python package installer)
- virtualenv or venv module

## Setup Instructions

1. **Create and activate a virtual environment**

   ```bash
   # Create virtual environment
   python -m venv venv

   # Activate virtual environment
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

2. **Install the package**

   With the virtual environment activated, install the package in editable mode:
   ```bash
   pip install -e .
   ```
   This will install the package along with all its dependencies as specified in setup.py.

## Important Notes
- Make sure you're using Python version between 3.9 and 3.11
- Always activate the virtual environment before running the project
- The setup.py file must be in the root directory of the project

## Troubleshooting
If you encounter any issues:
- Verify your Python version: `python --version`
- Ensure your virtual environment is activated
- Try upgrading pip: `pip install --upgrade pip`
- If installation fails, check for error messages in the console output

## Dependencies
All required dependencies will be automatically installed when running `pip install -e .`
