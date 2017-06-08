from setuptools import setup

setup(name="ccpr",
      version="0.1",
      description="A library for control chart pattern recognition",
      url="https://github.com/khayyamsaleem/ccpr",
      author="Khayyam Saleem",
      author_email="ksaleem@stevens.edu",
      license="MIT",
      packages=["ccpr"],
      install_requires=[
          "matplotlib",
          "numpy"
      ],
      zip_safe=False)
