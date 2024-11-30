from setuptools import setup, find_packages

setup(
    name="worker_scheduler",
    version="0.1.0",
    package_dir={"": "src"},  # tell setuptools packages are under src
    packages=find_packages(where="src"),  # look for packages under src
    install_requires=[
        'gym',
        'numpy',
        'pandas',
        'matplotlib',
        'torch',
    ]
)