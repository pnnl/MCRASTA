from setuptools import setup, find_packages
import platform

if platform.system() == 'Windows':
    requirements = ['pymc >= 5.5',
                    'arviz >= 0.15.1',
                    'pytensor >= 2.12.1',
                    'h5py >= 3.10.0',
                    'numpy >= 1.24.3',
                    'matplotlib',
                    'scipy >= 1.11.4',
                    'seaborn >= 0.12.2',
                    'pandas >= 2.0.2',
                    ]
elif platform.system() == 'linux-64':
    requirements = ['arviz >= 0.16.0',
                    'h5py >= 3.9.0'
                    'matplotlib-base >= 3.7.2'
                    'pandas >= 2.0.3'
                    'pymc >= 5.6.1'
                    'pytensor >= 2.12.3'
                    'python >= 3.11.4'
                    'scipy >= 1.11.1'
                    'seaborn >= 0.12.2'
                    ]
else:
    raise OSError('Unknown Operating System: {} {}'.format(platform.os.name, platform.system()))

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
        'Intended Audience :: Developers',
        'Natural Language :: English',
    ]
)
