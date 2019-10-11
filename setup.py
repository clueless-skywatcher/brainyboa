import setuptools
from setuptools import find_packages

setuptools.setup(
    name = 'brainyboa',
    version = '0.1',
    description = 'A Machine Learning Library implemented from scratch with Numpy',
    url = 'http://github.com/clueless-skywatcher/brainyboa',
    author = 'Somiparno Chattopadhyay',
    author_email = 'somichat@gmail.com',
    license = 'MIT',
    packages = find_packages(),
    zip_safe = False,
    install_requires = [
        'numpy'
    ]
)