import io
import os
import time
import re
import string
from PIL import Image, ImageFilter

import requests
import numpy as np
import pandas as pd

from scipy.fftpack import fft
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# import sklearn.metrics as sm

from keras.preprocessing import image
from keras.applications.inception_v3 \
    import decode_predictions, preprocess_input


class Unicorn:

    def __init__(self):
        self.dupe_images = True
        self.target_size = (299, 299)
        self.model = InceptionV3(weights='imagenet', include_top=False)
        self.alpha_fill = '#ffffff'
        self.prep_func = preprocess_input
        self.scale_features = True
        self.n_clusters = 4

    def load_image(self, img_path):
        ''' load image given path and convert to an array
        '''
        img = image.load_img(img_path, target_size=self.target_size)
        x = image.img_to_array(img)
        return self.prep_func(x)

    def load_image_from_web(self, image_url):
        ''' load an image from a provided hyperlink
        '''
        # get image
        response = requests.get(image_url)
        with Image.open(io.BytesIO(response.content)) as img:
            # fill transparency if needed
            if img.mode in ('RGBA', 'LA'):
                img = self.strip_alpha_channel(img)

            # convert to jpeg
            if img.format is not 'jpeg':
                img = img.convert('RGB')

            img.save('target_img.jpg')

    def validate_url(self, url):
        ''' takes input string and returns True if string is
            a url.
        '''
        url_validator = re.compile(
            r'^(?:http|ftp)s?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  #domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)

        return bool(url_validator.match(url))

    def featurize_image(self, image_array):
        ''' Returns binary array with ones where the model predicts that
            the image contains an instance of one of the target classes
            (specified by wordnet id)
        '''
        predictions = self.model.predict(image_array)

        return predictions

    def strip_alpha_channel(self, image):
        ''' Strip the alpha channel of an image and fill with fill color
        '''
        background = Image.new(image.mode[:-1], image.size, self.fill_color)
        background.paste(image, image.split()[-1])
        return background

    def cluster_images(self, image_paths):
        num_images = len(image_paths)
        feature_data = pd.DataFrame()

        for i, image_path in enumerate(image_paths):

            if self.validate_url(image_path):
                filename = 'target_img.jpg'
                self.load_image_from_web(image_path)
            else:
                filename = image_path

            if i % 10 == 0:
                print('processing image {}/{}'.format(i + 1, num_images))
            X = np.array([self.load_image(filename)])

            if self.dupe_images:

                # # # keras features
                image_features = self.featurize_image(X)

            else:
                # # # fourier transform only
                image_features = fft(X.flatten())

            if filename == 'target_img.jpg':
                os.remove('target_img.jpg')

            # # # keras with fourier transform
            # feature_data = feature_data.append(
            #     pd.Series(fft(image_features.flatten())),
            #     ignore_index=True)

            # # keras/fft only
            feature_data = feature_data.append(
                pd.Series(image_features.flatten()),
                ignore_index=True)

        feature_data['label'] = pd.Series(
            [i.split('/')[-1] for i in image_paths]
        )

        targets = feature_data[feature_data.columns.difference(['label'])]

        if self.scale_features:
            # # # standardize features? eg for PCA
            targets = StandardScaler().fit_transform(targets)

        # # # apply PCA feature matrix
        pca = PCA(n_components=3)
        pca_features = pca.fit_transform(
            targets
        )

        pca_feature_data = pd.DataFrame(
            data=pca_features,
            columns=['pc1', 'pc2', 'pc3'])

        model = KMeans(n_clusters=self.n_clusters)
        model.fit(
            pca_feature_data
        )

        feature_data['pred_class'] = pd.Series(model.labels_)

        return feature_data[['label', 'pred_class']]


if __name__ == '__main__':
    from keras.applications.inception_v3 import InceptionV3
    unicorn = Unicorn()
    # # # use fourier transform
    # unicorn.dupe_images = False
    # confirm URL points to a valid image when testing:
    sample_data = [(os.getcwd() + '/images/' + i) for i in os.listdir('images')]
    result = unicorn.cluster_images(sample_data)
    print(result)
