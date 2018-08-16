from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image 

# Torch
import torch
from torchvision.transforms import transforms

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# argparse for setup
import argparse

parser = argparse.ArgumentParser(description='Deploy an Image Classifier')
parser.add_argument('-k', '--keras-model', metavar='PATH', help='path to the keras model to use', type=str)
parser.add_argument('-t', '--torch-model', metavar='PATH', 
help='path to the keras model to use(won\'t work along with -k or --keras)', type=str)
parser.add_argument('-d', '--img-dim', metavar=('H','W'), 
help='specify image dimensions as height(H) and width(W)', nargs=2, default=[224, 224], type=int)
parser.add_argument('-c', '--classes', metavar='PATH', help='path to the text file that have class names', type=str)
args = parser.parse_args()

print(args)

# default model type
model_type = 'keras'

if args.keras_model != None:
    MODEL_PATH = args.keras_model
    model_type = 'keras'

    # Load your trained model
    model = load_model(MODEL_PATH)
    model._make_predict_function() # Necessary
    print('Model loaded. Start serving...')
elif args.torch_model != None:
    MODEL_PATH = args.torch_model
    model_type = 'torch'

    model = torch.load(MODEL_PATH, map_location='cpu')
else: 
    # You can also use pretrained model from Keras
    # Check https://keras.io/applications/
    from keras.applications.resnet50 import ResNet50
    model = ResNet50(weights='imagenet')


# Define a flask app
app = Flask(__name__)
print('Model loaded. Check http://127.0.0.1:5000/')


def preprocess_for_keras(img):
    # Preprocessing the image
    img = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    img = np.expand_dims(img, axis=0)
    return img


def preprocess_for_torch(img):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(112),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = transform(img)
    img = img.unsqueeze(0)
    return img


def preprocess(img, model_type):
    
    # preprocess image as per the type of model
    if model_type == 'keras':
        return preprocess_for_keras(img)
    else:
        return preprocess_for_torch(img)


def model_predict(img_path, model_type, model):
    # H and W
    height = args.img_dim[0]
    weight = args.img_dim[1]

    # load a PIL Image
    img = image.load_img(img_path, target_size=(height, weight))
    img = preprocess(img, model_type)

    if model_type == 'keras':
        # Be careful how your trained model deals with the input
        # otherwise, it won't make correct prediction!
        img = preprocess_input(img, mode='caffe')
        preds = model.predict(img)
    elif model_type == 'torch':
        output = model(img)
        _, preds = torch.argmax(output, axis=1)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model_type, model)
        print(preds)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        result = str(pred_class[0][0][1])               # Convert to string
        return result
    return None


if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
