from setuptools import setup


setup(name='nk_unicorn',
      version='1.0.0',
      description='UNsupervised Image Clustering with Object Recognition Network system',
      packages=['nk_croc'],
      install_requires=[
          'tensorflow == 1.8.0',
          'Keras == 2.1.6',
          'pandas >= 0.22.0, <= 0.23.0',
          'numpy >= 1.13.3',
          'Pillow >= 5.1.0'],
      )
