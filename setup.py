from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='bces',
    version='1.6',
    description='Python module for performing linear regression for data with measurement errors and intrinsic scatter',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='Rodrigo Nemmen',
    author_email='rodrigo.nemmen@iag.usp.br',
    url='https://github.com/rsnemmen/BCES',
    download_url = 'https://github.com/rsnemmen/BCES/archive/1.6.tar.gz',
    license=license,
    keywords = ['statistics', 'fitting', 'linear-regression','machine-learning'],
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=['numpy', 'scipy', 'tqdm'],
    python_requires='>=3.8',
)