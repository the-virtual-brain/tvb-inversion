from setuptools import find_packages, setup

TVB_INVERSION_VERSION = '0.3.0'
SBI_TVB_REQUIREMENTS = ["autopep8", "sbi", "scipy", "torch", "numpy", "tvb-library", "dask", "distributed", "pandas",
                        "pymc3"]
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
