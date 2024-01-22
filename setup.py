from setuptools import setup, find_packages

setup(
    name='pyRLFramework',
    version='0.1.0',
    packages=find_packages(include=['RLFramework', 'RLFramework.*']),
    install_requires=[
        'numpy',
        'torch'
    ]
)
