from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='bces',
    version='0.5',
    description='Python module for performing linear regression for data with measurement errors and intrinsic scatter',
    long_description=readme,
    author='Rodrigo Nemmen',
    author_email='rodrigo.nemmen@iag.usp.br',
    url='https://github.com/rsnemmen/bces',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)