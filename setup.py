from setuptools import find_packages, setup

TVB_INVERSION_VERSION = '1.0.0'
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
    download_url='https://github.com/the-virtual-brain/tvb-inversion',
)
