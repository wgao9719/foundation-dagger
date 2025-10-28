from setuptools import setup, find_packages


def parse_requirements(filename):
    with open(filename, "r") as f:
        return f.read().splitlines()


setup(
    name="dfot",
    version="0.0.1",
    description="",
    packages=find_packages(),
    install_requires=parse_requirements("./requirements.txt"),
)
