from setuptools import setup, find_packages

setup(
    name='neurotopo-alpha',
    version='0.1',
    description='Topological deep learning framework for alpha inference',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'torch',
        'gudhi',
        'pywt',
        'matplotlib'
    ],
)
