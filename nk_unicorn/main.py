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
import sklearn.metrics as sm
from keras.preprocessing import image
from keras.applications.inception_v3 \
    import decode_predictions, preprocess_input


def load_image(img_path, target_size, prep_func=lambda x: x):
    ''' load image given path and convert to an array
    '''
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)
    return prep_func(x)


def load_image_from_web(image_url):
    ''' load an image from a provided hyperlink
    '''
    # get image
    response = requests.get(image_url)
    with Image.open(io.BytesIO(response.content)) as img:
        # fill transparency if needed
        if img.mode in ('RGBA', 'LA'):
            img = strip_alpha_channel(img)

        # convert to jpeg
        if img.format is not 'jpeg':
            img = img.convert('RGB')

        img.save('target_img.jpg')


def validate_url(url):
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


def featurize_image(image_array, model):
    ''' Returns binary array with ones where the model predicts that
        the image contains an instance of one of the target classes
        (specified by wordnet id)
    '''
    predictions = model.predict(image_array)

    return predictions


def strip_alpha_channel(image, fill_color='#ffffff'):
    ''' Strip the alpha channel of an image and fill with fill color
    '''
    background = Image.new(image.mode[:-1], image.size, fill_color)
    background.paste(image, image.split()[-1])
    return background


def unicorn(image_paths, model, target_size=(299, 299)):
    num_images = len(image_paths)
    feature_data = pd.DataFrame()

    for i, image_path in enumerate(image_paths):

        if validate_url(image_path):
            filename = 'target_img.jpg'
            load_image_from_web(image_path)
        else:
            filename = image_path

        print('processing image {}/{}'.format(i + 1, num_images))
        X = np.array(
            [load_image(
                filename, target_size, prep_func=preprocess_input)])

        # print('extracting image features')
        # image_features = featurize_image(X, model)

        # # # fourier transform only
        print('extracting image features')
        print(X.shape)
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

    model = KMeans(n_clusters=4)
    model.fit(
        feature_data[feature_data.columns.difference(['label'])]
    )

    print(feature_data.shape)

    feature_data['pred_class'] = pd.Series(model.labels_)

    return feature_data[['label', 'pred_class']]


if __name__ == '__main__':
    from keras.applications.inception_v3 import InceptionV3
    model = InceptionV3(weights='imagenet', include_top=False)
    # confirm URL points to a valid image when testing:
    result = unicorn([(os.getcwd() + '/images/' + i) for i in os.listdir('images')], model)
    print(result)

