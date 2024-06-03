from setuptools import setup, find_packages

with open('requirements_windows.txt') as requirements_file:
    requirements = requirements_file.read()

setup(
    name='mcrasta',
    version='0.1.0',
    description='Markov chain Monte Carlo + rate & state friction modeler',
    url='https://github.com/pnnl/MCRASTA',
    author='Marissa Fichera',
    author_email='marissa.fichera@pnnl.gov',
    license='BSD license',
    packages=find_packages(include=['mcrasta', 'mcrasta.*']),
    install_requires=requirements,

    classifiers=[
        'Development Status :: ',
        'Intended Audience :: Developers',
        'License :: unsure :: unsure',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
    ]
)
