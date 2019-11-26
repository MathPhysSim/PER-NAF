import os

from setuptools import setup, find_packages

try:
    from pypandoc import convert, convert_text


    def read_md(file_path):
        return convert_text(file_path, to='rst', format='markdown')

except ImportError:
    convert = None
    print(
        "warning: pypandoc module not found, "
        "could not convert Asciidoc to RST"
    )
    def read_md(file_path):
         with open(file_path, 'r') as f:
            return f.read()

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


# README = os.path.join(os.path.dirname(__file__), 'README.md')


def strip_comments(l):
    return l.split('#', 1)[0].strip()


def reqs(*f):
    return list(filter(None, [strip_comments(l) for l in open(
        os.path.join(os.getcwd(), *f)).readlines()]))

setup(
    name='pernaf',
    version="0.0.14",
    author="Simon Hirlaender",
    author_email="simon.hirlaender@cern.ch",
    description="An implementation of the Normalized Advantage Function Reinforcement Learning Algorithm with "
                "Prioritized Experience Replay",
    url = 'https://github.com/MathPhysSim/PER-NAF',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=['tensorflow==1.14', 'numpy', 'gym'],
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6'
)