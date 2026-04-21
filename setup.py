from setuptools import setup, find_packages

setup(
    name='brainscore',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'brainscore_core',
        'brainscore_vision',
        'brainscore_language',
    ],
)
