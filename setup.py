from setuptools import setup, find_packages

setup(
    name="boltzmann",
    author="Justin L. MacCallum",
    author_email="justin.maccallum@ucalgary.ca",
    packages=find_packages(),
    url="http://maccallumlab.org",
    license="LICENSE.txt",
    description="Boltzmann Generators implemented in PyTorch using Neural Spline Flows",
    long_description=open("README.md").read(),
)
