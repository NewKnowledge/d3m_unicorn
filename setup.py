from setuptools import setup


setup(name='d3m_unicorn',
      version='1.0.0',
      description='UNsupervised Image Clustering with Object Recognition Network system',
      packages=['d3m_unicorn'],
      install_requires=[
          'tensorflow == 1.8.0',
          'Keras == 2.1.6',
          'numpy >= 1.13.3',
          'pandas >= 0.22.0, <= 0.23.0',
          'Pillow >= 5.1.0',
          'PyWavelets==0.5.2',
          'scipy==1.1.0',
          'scikit-learn==0.19.1'],
      )
