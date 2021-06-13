import os
from setuptools import setup


PRODUCT_NAME = 'modlr'
DEPENDENCIES = [
    'azure-storage-blob==12.8.1',
    'azure-identity==1.4.1',
    'numerapi==2.4.0',
    'numpy==1.19.5',
    'pandas==1.2.4',
    'rich==9.10.0',
    'scikit-learn==0.24.2',
    'scipy==1.6.3'
]

VERSION = os.environ.get('GITVERSION_MAJORMINORPATCH', '0.0.3')

setup(
    name=PRODUCT_NAME,
    version=VERSION,
    author='chelnak',
    description='A small framework to help with Numerai modeling',
    url='https://github.com/chelnak/modlr',
    packages=['modlr'],
    install_requires=DEPENDENCIES,
)