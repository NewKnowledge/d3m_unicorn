from setuptools import setup


setup(name='d3m_unicorn',
      version='1.0.0',
      description='UNsupervised Image Clustering with Object Recognition Network system',
      packages=['d3m_unicorn'],
      install_requires=[
          'tensorflow-gpu <= 1.12.0',
          'Keras <= 2.2.4, >=2.1.6',
          'numpy >= 1.15.4',
          'pandas == 0.23.4',
          'Pillow >= 5.1.0',
          'PyWavelets==0.5.2',
          'scipy==1.1.0',
          'scikit-learn==0.20.3'],
      )
