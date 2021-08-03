from setuptools import find_packages, setup

REQUIREMENTS = ["julia",
                "matplotlib",
                "numba",
                "numpy",
                "sbi",
                "scikit-learn",
                "scipy",
                "torch"]
setup(
    name='sbi-julia',
    packages=find_packages(),
    version='0.1.0',
    install_requires=REQUIREMENTS,
)
