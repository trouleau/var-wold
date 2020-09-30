from distutils.core import setup
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name='lib',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='0.0.1',

    description='Algorithms for multivariates point processes',
    long_description='Algorithms for multivariates point processes',

    # The project's main homepage.
    url='',

    # Author details
    author='',
    author_email='',

    # Choose your license
    license='MIT',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
    keywords='Multivariate Point Processes',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    install_requires=[
        "certifi==2019.11.28",
        "numpy==1.18.1",
        "torch==1.3.1",
        "scipy==1.4.1",
        "numba==0.47.0",
        "pandas==0.25.3",
        "matplotlib==3.1.2",
        "networkx==2.2"
    ],
    extras_require={},
    package_data={},
    data_files=[],
    entry_points={},
)
