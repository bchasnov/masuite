"""Install script for setuptools."""

import imp

import setuptools

setuptools.setup(
    name='masuite',
    description=(''),
    long_description=open('readme.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/bchasnov/masuite',
    author='',
    author_email='',
    license='',
    version=imp.load_source('_metadata', 'masuite/_metadata.py').__version__,
    keywords='',
    packages=setuptools.find_packages(),
    install_requires=[
        'absl-py',
        'frozendict',
        #'gym',
        'matplotlib',
        'numpy',
        'pandas',
        'plotnine',
        'scipy',
        'termcolor',
        'torch',
        'ipykernel',
        'jax',
        'jaxlib',
        'pytest'
    ],
)
