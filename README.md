# d3m_unicorn
UNsupervised Image Clustering with Object Recognition Network

## Quick Start

Use UNICORN in your project via pip with `pip3 install -e <path/to/unicorn>`.

or

Start UNICORN as a service on your local-machine with:

1) `docker build -t unicorn-http:dev -f ./http.dockerfile .`
2) `docker run -p 5000:5000 unicorn-http:dev`

## Structure of this repo

The core of this repo is `setup.py` and `nk_unicorn`. 

This repo is pip-installsable and makes the contents of `nk_unicorn` available after installation.

There is a flask wrapper for the library located in `http-wrapper`. It uses `nk_unicorn` and can be built with the `http.dockerfile`. For more information see [the README.md in `http-wrapper`](./http-wrapper/README.md)

## Coming soon

agglomerative clustering on InceptionV3 Imagenet features with T-SNE dimensionality reduction
