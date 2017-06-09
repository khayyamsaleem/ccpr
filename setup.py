from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name="ccpr",
      version="0.1",
      description="A library for control chart pattern recognition",
      long_description=readme(),
      url="https://github.com/khayyamsaleem/ccpr",
      author="Khayyam Saleem",
      author_email="ksaleem@stevens.edu",
      license="MIT",
      packages=["ccpr"],
      scripts=['ccpr/bin/ccpr'],
      install_requires=[
          "matplotlib",
          "numpy"
      ],
      zip_safe=False)
