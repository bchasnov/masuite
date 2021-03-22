"""Install script for setuptools."""

import imp

import setuptools

# # Additional requirements for TensorFlow baselines, excluding OpenAI & Dopamine.
# # See baselines/README.md for more information.
# baselines_require = [
#     'dm-sonnet',
#     'dm-tree',
#     'tensorflow',
#     'tensorflow_probability',
#     'trfl',
#     'tqdm',
# ]

# # Additional requirements for JAX baselines.
# # See baselines/README.md for more information.
# baselines_jax_require = [
#     'dataclasses',
#     'dm-haiku',
#     'dm-tree',
#     'jax',
#     'jaxlib',
#     'optax',
#     'rlax',
#     'tqdm',
# ]

# baselines_third_party_require = [
#     'tensorflow == 1.15',
#     'dopamine-rl',
#     'baselines',
# ]

# testing_require = [
#     'mock',
#     'pytest-xdist',
#     'pytype',
# ]

setuptools.setup(
    name='masuite',
    description=(''),
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/bchasnov/masuite',
    author='',
    author_email='',
    license='',
    version=imp.load_source('_metadata', 'bsuite/_metadata.py').__version__,
    keywords='',
    packages=setuptools.find_packages(),
    install_requires=[
#         'absl-py',
#         'dm_env',
#         'frozendict',
        'gym',
        'matplotlib',
        'numpy',
        'pandas',
        'plotnine',
        'scipy',
#         'scikit-image',
#         'six',
        'termcolor',
    ],
#     extras_require={
#         'baselines': baselines_require,
#         'baselines_jax': baselines_jax_require,
#         'baselines_third_party': baselines_third_party_require,
#         'testing': testing_require,
#     },
#     classifiers=[ ],
)
