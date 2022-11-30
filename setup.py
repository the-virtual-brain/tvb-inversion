from setuptools import find_packages, setup

TVB_INVERSION_VERSION = '0.4.0'
SBI_TVB_REQUIREMENTS = ["autopep8", "sbi", "scipy", "torch", "numpy", "tvb-library", "dask", "distributed", "pandas"]
TVB_INVERSION_TEAM = "Jan Fousek, Meysam Hashemi, Abolfazl Ziaee Mehr"

setup(
    name='tvb-inversion',
    packages=find_packages(),
    version=TVB_INVERSION_VERSION,
    install_requires=SBI_TVB_REQUIREMENTS,
    author=TVB_INVERSION_TEAM,
    author_email='tvb.admin@thevirtualbrain.org',
    url='http://www.thevirtualbrain.org',
    license="GPL-3.0-or-later",
    download_url='https://github.com/the-virtual-brain/tvb-inversion',
)
