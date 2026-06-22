from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name = 'ESG',
    version= 0.1,
    author= 'Pooja',
    packages= find_packages(),
    install_requires= requirements
)
