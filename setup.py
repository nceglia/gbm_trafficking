from setuptools import setup, find_packages

setup(
    name='trafficking',
    version='0.0.1',
    description='GBM Trafficking',
    packages=find_packages(include=['trafficking','trafficking.model','trafficking.inference','trafficking.plotting','trafficking.data']),
)
