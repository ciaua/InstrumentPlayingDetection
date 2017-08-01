from setuptools import setup, find_packages
import jjtorch


setup(name='InstrumentPlayingDetection',
      version=jjtorch.__version__,
      description='Codes for visual instrument detection',
      author='Jen-Yu Liu',
      author_email='ciaua@citi.sinica.edu.tw',
      license='ISC',
      packages=find_packages(),
      )
