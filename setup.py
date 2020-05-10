#!/usr/bin/env python

from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='p2_udacity_rl_multiagents',
    version='1.0',
    url='https://github.com/jensakut/Reinforcement-Learning-P2-Multiagents',
    license='MIT',
    author='Jens Kutschera',
    author_email='',
    description='',
    packages=find_packages(),
    install_requires=required
)
