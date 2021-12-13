try:
    from setuptools import setup, find_packages
except ImportError:
    from ez_setup import use_setuptools
    use_setuptools()
    from setuptools import setup, find_packages


__version__ = '0.1.0'

setup(
    name='photonai_neuro',
    packages=find_packages(),
    include_package_data=True,
    version=__version__,
    description="""
PHOTONAI NEURO
The Neuro Module enables loading and preprocessing neuroimaging data such as structural and 
functional Magnetic Resonance Imaging (MRI) data. 
In addition, it supports a range of advanced feature extraction and feature engineering as well as atlas-based analyses. 
""",
    author='PHOTONAI Team',
    author_email='hahnt@wwu.de',
    url='https://github.com/mmll-wwu/photonai_neuro.git',
    download_url='https://github.com/wwu-mmll/photonai_neuro/archive/' + __version__ + '.tar.gz',
    keywords=['machine learning', 'neuroimaging', 'MRI'],
    classifiers=[],
    install_requires=[
        'photonai',
        'nibabel',
        'nilearn',
        'scikit-image']
)
