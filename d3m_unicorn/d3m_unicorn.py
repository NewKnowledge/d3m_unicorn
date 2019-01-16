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
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# import sklearn.metrics as sm

from keras.preprocessing import image
from keras.applications.inception_v3 \
    import decode_predictions, preprocess_input
from keras.applications.inception_v3 import InceptionV3


class Unicorn():

    def __init__(self, weights_path):
        self.cnn_features = True
        self.target_size = (299, 299)
        self.alpha_fill = '#ffffff'
        self.prep_func = preprocess_input
        self.scale_features = True
        self.n_clusters = 4
        self.n_pca_comps = 10
        self.model = InceptionV3(weights=weights_path)

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
        background = Image.new(image.mode[:-1], image.size, self.alpha_fill)
        background.paste(image, image.split()[-1])
        return background

    def fft_images(self, image_paths):
        ''' Returns the fft transform of images from paths provided as a list
        '''
        num_images = len(image_paths)
        feature_data = pd.DataFrame()

        for i, image_path in enumerate(image_paths):

            try:
                if self.validate_url(image_path):
                    filename = 'target_img.jpg'
                    self.load_image_from_web(image_path)
                else:
                    filename = image_path

                if i % 10 == 0:
                    print('processing image {}/{}'.format(i + 1, num_images))
                X = np.array([self.load_image(filename)])

                # # # flatten and apply fft
                image_features = fft(X.flatten())

                if filename == 'target_img.jpg':
                    os.remove('target_img.jpg')

                feature_data = feature_data.append(
                    pd.Series(image_features),
                    ignore_index=True)

                # feature_data = feature_data.append(
                #     pd.Series(image_features.flatten()),
                #     ignore_index=True)

            except Exception as e:
                print(e)
                feature_data = feature_data.append(
                    pd.Series([np.nan]),
                    ignore_index=True)

        feature_data = feature_data.set_index(
            pd.Series(
                [i.split('/')[-1] for i in image_paths]
            )
        )

        return feature_data

    def get_net_features(self, image_paths):
        ''' Returns features of images (defaults to inception V3:imagenet wts)
            from paths provided as a list
        '''
        # from keras.applications.inception_v3 import InceptionV3
        self.model = InceptionV3(weights='imagenet', include_top=False)
        num_images = len(image_paths)
        feature_data = pd.DataFrame()

        for i, image_path in enumerate(image_paths):

            try:
                if self.validate_url(image_path):
                    filename = 'target_img.jpg'
                    self.load_image_from_web(image_path)
                else:
                    filename = image_path

                if i % 10 == 0:
                    print('processing image {}/{}'.format(i + 1, num_images))
                X = np.array([self.load_image(filename)])

                # # # flatten and get
                image_features = self.featurize_image(X)

                if filename == 'target_img.jpg':
                    os.remove('target_img.jpg')

                feature_data = feature_data.append(
                    pd.Series(image_features.flatten()),
                    ignore_index=True)

            except Exception as e:
                print(e)
                feature_data = feature_data.append(
                    pd.Series([0]),
                    ignore_index=True)

        feature_data = feature_data.set_index(
            pd.Series(
                [i.split('/')[-1] for i in image_paths]
            )
        )

        feature_data = feature_data.dropna(how='any')

        return feature_data

    def haar_wavelet_features():
        pass

    def pca_image_features(self, feature_data):
        ''' Runs PCA on images in dataframe where rows are images
            and columns are features.
        '''
        if self.scale_features:
            # # # standardize features? eg for PCA
            feature_data = pd.DataFrame(
                data=StandardScaler().fit_transform(feature_data)
            )

        # # # apply PCA feature matrix
        pca = PCA(n_components=self.n_pca_comps)
        pca_features = pca.fit_transform(
            feature_data
        )

        pca_feature_data = pd.DataFrame(
            data=pca_features,
            columns=['pc' + str(i) for i in range(0, self.n_pca_comps)])
        pca_feature_data.set_index(feature_data.index)

        return pca_feature_data

    def calc_distance(self, feature_data):
        ''' Calculate pairwise feature distance between images in a dataframe
            where rows are images and columns are features
        '''
        from scipy.spatial.distance import squareform, pdist

        pwise_dist_df = pd.DataFrame(
            squareform(
                pdist(feature_data)
            ),
            columns=feature_data.index,
            index=feature_data.index
        )

        return pwise_dist_df

    def run_kmeans(self, feature_data, target_data):

        model = KMeans(n_clusters=self.n_clusters)
        model.fit(target_data)

        output_data = pd.concat(
            {'label': pd.Series(feature_data.index),
             'pred_class': pd.Series(model.labels_)},
            axis=1
        )

        return output_data

    def run_knn(self, target_data):

        nbrs = NearestNeighbors(
            n_neighbors=2,
            algorithm='ball_tree'
        ).fit(target_data)
        distances, indices = nbrs.kneighbors(target_data)

        output_data = pd.concat(
            {'indices_0': pd.Series(target_data.index[indices[:, 0]]),
             'indices_1': pd.Series(target_data.index[indices[:, 1]]),
             'distances': pd.Series(distances[:, 1])},
            axis=1
        ).sort_values('distances')

        return output_data

    def cluster_images(self, image_paths):

        # Use CNN-generated features
        if self.cnn_features:
            feature_data = self.get_net_features(image_paths)

            # # kmeans on imagenet activation
            # processed_feature_data = self.pca_image_features(feature_data)
            # result = self.run_kmeans(feature_data, processed_feature_data)

            # # kmeans on pairwise distance
            pwise_dist_df = self.calc_distance(feature_data)
            result = self.run_kmeans(feature_data, pwise_dist_df)

            # knn on image features
            # result = self.run_knn(feature_data)

        # Use fast fourier transform
        else:
            feature_data = self.fft_images(image_paths)
            processed_feature_data = self.pca_image_features(feature_data)

            result = self.run_kmeans(feature_data, processed_feature_data)

        return result


if __name__ == '__main__':
    unicorn = Unicorn()

    # # # use fourier transform
    # unicorn.cnn_features = False

    # confirm sample_data grabs valid image paths when testing:
    sample_data = [(os.getcwd() + '/images/' + i) for i in os.listdir('images')]
    result = unicorn.cluster_images(sample_data)
    print(result)
