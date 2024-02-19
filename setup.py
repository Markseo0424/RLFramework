from setuptools import setup, find_packages

setup(
    name='RLFramework',
    version='0.2.7',
    packages=find_packages(include=['RLFramework', 'RLFramework.*']),
    url="https://github.com/Markseo0424/RLFramework.git",
    install_requires=[
        'numpy',
        'torch'
    ]
)
