import os
from setuptools import setup, find_packages


long_description = (
    "xmu is a Python utility used to read and write XML for Axiell EMu,"
    " a collections management system used in museums, galleries, and"
    " similar institutions."
    "\n\n"
    " Install with:"
    "\n\n"
    "```\n"
    "pip install xmu\n"
    "```"
    "\n\n"
    "Learn more:\n\n"
    "+ [GitHub repsository](https://github.com/NMNH-IDSC/xmu)\n"
    "+ [Documentation](https://xmu.readthedocs.io/en/latest/)"
)


setup(
    name="xmu",
    maintainer="Adam Mansur",
    maintainer_email="mansura@si.edu",
    description="Reads and writes XML for Axiell EMu",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version="0.2b1",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Topic :: Text Processing :: Markup :: XML",
    ],
    url="https://github.com/NMNH-IDSC/xmu.git",
    license="MIT",
    packages=find_packages(),
    install_requires=["joblib", "lxml", "pyyaml", "requests"],
    include_package_data=True,
    zip_safe=False,
)
