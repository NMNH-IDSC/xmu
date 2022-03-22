import os
from setuptools import setup, find_packages


long_description = (
    "xmu is a Python utility used to read and write XML for Axiell EMu,"
    " a collections management system used in museums, galleries, and"
    " similar institutions.\n\n"
    "Learn more:\n\n"
    "+ [GitHub repsository](https://github.com/adamancer/xmu)"
)


setup(
    name="xmu",
    maintainer="Adam Mansur",
    maintainer_email="mansura@si.edu",
    description="Reads and writes XML for Axiell EMu",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version="0.1b1",
    classifiers=[
        "License :: OSI Approved :: MIT License",
    ],
    url="https://github.com/adamancer/xmu.git",
    license="MIT",
    packages=find_packages(),
    install_requires=["lxml", "pyyaml"],
    include_package_data=True,
    zip_safe=False,
)
