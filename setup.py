import pathlib
import re
import setuptools


def parse_reqs(filename):
  requirements = pathlib.Path(filename)
  requirements = requirements.read_text().split('\n')
  requirements = [x for x in requirements if x.strip()]
  return requirements


def parse_version(filename):
  text = (pathlib.Path(__file__).parent / filename).read_text()
  version = re.search(r"__version__ = '(.*)'", text).group(1)
  return version


setuptools.setup(
    name='embodied',
    version=parse_version('embodied/__init__.py'),
    author='Danijar Hafner',
    author_email='mail@danijar.com',
    description='Fast reinforcement learning research',
    url='http://github.com/danijar/embodied',
    long_description=pathlib.Path('README.md').read_text(),
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=parse_reqs('embodied/requirements.txt'),
    extras_require={
        'dreamerv3': parse_reqs('embodied/agents/dreamerv3/requirements.txt'),
    },
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
