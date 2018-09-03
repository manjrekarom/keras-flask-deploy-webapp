from __future__ import division, print_function
# coding=utf-8
import sys, csv, os, glob, time, re
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

# argparse
import argparse

# setup for argparse
parser = argparse.ArgumentParser(description='Deployan Image Classifier')
parser.add_argument('-p', '--port', metavar='PORT', help='port for web application', type=int)
parser.add_argument('-k', '--keras-model', metavar='PATH', help='path to the keras model to use', type=str)
parser.add_argument('-t', '--torch-model', metavar='PATH', 
help='path to the keras model to use(won\'t work along with -k or --keras)', type=str)
parser.add_argument('-d', '--img-dim', metavar=('H','W'), 
help='specify image dimensions as height(H) and width(W)', nargs=2, default=[224, 224], type=int)
parser.add_argument('-c', '--classes', metavar='PATH', help='path to the text file that have class names', type=str)
parser.add_argument('--delim', metavar='DELIMETER', help='delimeter for file categories', type=str)
args = parser.parse_args()

print(args)
# Global variables
MODEL_TYPES = {"k": "KERAS", "t": "TORCH", "kp": "KERAS_PRETRAINED"}
CLASSES = None


if args.keras_model != None:
    MODEL_PATH = args.keras_model
    model_type = MODEL_TYPES["k"]
    # Load your trained model
    model = load_model(MODEL_PATH)
    model._make_predict_function() # Necessary
    print('Keras model loaded. Start serving...')

elif args.torch_model != None:
    MODEL_PATH = args.torch_model
    model_type = MODEL_TYPES["t"]
    model = torch.load(MODEL_PATH, map_location='cpu')
    print('Torch model loaded. Start serving...')

else: 
    # You can also use pretrained model from Keras
    # Check https://keras.io/applications/
    from keras.applications.resnet50 import ResNet50
    model = ResNet50(weights='imagenet')
    model_type = MODEL_TYPES["kp"]
    print('Pretrained keras model loaded. Start serving...')


# Define a flask app
app = Flask(__name__)
app.debug = True

print('Model loaded. Check http://127.0.0.1:' + str(args.port))


def preprocess_for_keras(img_path, img_size=(224, 224)):
    img = image.load_img(img_path, target_size=img_size)
    # Preprocessing the image
    img = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    img = np.expand_dims(img, axis=0)
    return img


def preprocess_for_torch(img_path, img_size=(224, 224)):
    img = image.load_img(img_path)
    # compose transforms
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = transform(img)
    img = img.unsqueeze(0)
    return img


def preprocess(img_path, model_type, img_size=(224, 224)):
    # preprocess image as per the type of model
    if model_type == MODEL_TYPES["k"] or model_type == MODEL_TYPES["kp"]:
        return preprocess_for_keras(img_path, img_size)
    else:
        return preprocess_for_torch(img_path, img_size)


def model_predict(img_path, model_type, model):
    # H and W
    height = args.img_dim[0]
    width = args.img_dim[1]

    if model_type == MODEL_TYPES["k"] or model_type == MODEL_TYPES["kp"]:
        # loads and preprocesses
        img = preprocess(img_path, model_type, img_size=(height, width))
        # Be careful how your trained model deals with the input
        # otherwise, it won't make correct prediction!
        img = preprocess_input(img, mode='caffe')
        preds = model.predict(img)

    elif model_type == MODEL_TYPES["t"]:
        # loads, preprocesses and converts to tensor
        img = preprocess(img_path, model_type, img_size=(height, width))
        preds = model(img)
    
    return preds


def get_classes(file_path, delimeter=','):
    classes = []
    with open(file_path) as file:
        for l in file:
            [classes.append(item) for item in l.strip().split(delimeter)]
    return classes


def human_level_predict(preds, model_type, top=1):
    # generate categories
    if args.classes != None:
        if args.delim != None:
            CLASSES = get_classes(args.classes, args.delim)
        else:
            CLASSES = get_classes(args.classes)
    
    if model_type == MODEL_TYPES["kp"]:
        # TODO: To be tested
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        pred_class = decode_predictions(preds, top=top)   # ImageNet Decode
        result = str(pred_class[0][0][1])               # Convert to string
    
    elif model_type == MODEL_TYPES["k"]:
        # TODO: Need to find what is the most common type of 
        # returned output and how to label it then
        if args.classes == None:
            CLASSES = list(range(preds.shape[1]))
        
        # find pos of max score
        preds = np.argmax(preds, axis=1)
        result = str(CLASSES[preds])

    elif model_type == MODEL_TYPES["t"]:
        if args.classes == None:
            CLASSES = list(range(preds.size(1)))
        
        # find pos of maximum score
        _, preds = torch.max(preds, dim=1)
        result = str(CLASSES[preds])

    return result


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

        # Process your result for human
        result = human_level_predict(preds, model_type)
        return result
    return None


if __name__ == '__main__':
    # app.run(port=5002, debug=True)
    # Serve the app with gevent
    if args.port == None:
       args.port = 5000
    http_server = WSGIServer(('', args.port), app)
    http_server.serve_forever()
