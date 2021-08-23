from setuptools import find_packages, setup

SBI_TVB_REQUIREMENTS = ["autopep8", "sbi", "scipy", "torch", "numpy"]

setup(
    name='sbi-tvb',
    packages=find_packages(),
    version='0.1.0',
    install_requires=SBI_TVB_REQUIREMENTS,
)
