import os

from setuptools import setup

try:
    from pypandoc import convert

    def read_md(file_path):
        return convert(file_path, to='rst', format='asciidoc')

except ImportError:
    convert = None
    print(
        "warning: pypandoc module not found, "
        "could not convert Asciidoc to RST"
    )

    def read_md(file_path):
         with open(file_path, 'r') as f:
            return f.read()

README = os.path.join(os.path.dirname(__file__), 'README')


def strip_comments(l):
    return l.split('#', 1)[0].strip()


def reqs(*f):
    return list(filter(None, [strip_comments(l) for l in open(
        os.path.join(os.getcwd(), *f)).readlines()]))

setup(
    name='per-naf',
    version="0.0.1",
    description="An implementation of the Normalized Advantage Function Reinforcement Learning Algorithm with Prioritized Experience Replay",
    long_description=read_md(README),
    long_description_content_type='text/x-rst',
    install_requires=reqs('requirements.txt')
)