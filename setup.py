import setuptools
import pathlib

requirements = pathlib.Path('embodied/requirements.txt')
requirements = requirements.read_text().split('\n')
requirements = [x for x in requirements if x.strip()]


setuptools.setup(
    name='embodied',
    version='1.0.1',
    author='Danijar Hafner',
    author_email='mail@danijar.com',
    description='Fast reinforcement learning research',
    url='http://github.com/danijar/embodied',
    long_description=pathlib.Path('README.md').read_text(),
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=requirements,
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
