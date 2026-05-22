from setuptools import setup,find_packages

import os
import sys
import platform

setup(name='minihf',
    version='0.1',
    description='MiniHF: A minimal pedagogical implementation of Hartree-Fock calculation from scratch.',
    url='https://zhaoyilin.github.io/minihf/',
    author='Yilin Zhao',
    author_email='zhaoyilin1991@gmail.com',
    license='MIT',
    packages=find_packages()
    )    