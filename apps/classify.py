import requests
import time
import os
import io
import sys

from flask_restful import Resource, Api, reqparse
from flask import request
from werkzeug import secure_filename
from apps import app
from apps.image_resize import ImageResize
from apps.classification import Tensorflow

api = Api(app)

class Classify(Resource):
    def __init__(self):
        self.reqparse = reqparse.RequestParser()

        self.reqparse.add_argument('graph',
            type = str,
            required = True,
            help = '',
            location = 'form')

        self.reqparse.add_argument('imageUrl',
            type = str,
            required = False,
            help = 'Image Url to download',
            location = 'form')

        self.reqparse.add_argument('imageCrop',
            type = str,
            required = False,
            default = "none",
            help = 'right | left | none',
            location = 'form')

        super(Classify, self).__init__()

    def post(self):
        postStart = time.time()

        args = self.reqparse.parse_args()

        start = time.time()
        if (args.imageUrl):
            response = requests.get(args.imageUrl, stream=True)
            image = io.BytesIO(response.content)
        else:
            image = request.files['image']
            image.seek(0)
        downloadTimeElapsed = (time.time() - start)

        start = time.time()
        imgResize = ImageResize()
        imgOutputPath = imgResize.resize(image, args.imageCrop.lower())
        resizeTimeElapsed = (time.time() - start)

        start = time.time()
        tf = Tensorflow()
        output = tf.execute(args.graph, imgOutputPath)
        classifyTimeElapsed = (time.time() - start)

        os.remove(imgOutputPath)

        postEnd = time.time()

        return {
            'totalTimeElapsedMs': (postEnd - postStart) * 1000,
            'downloadTimeElapsedMs': downloadTimeElapsed * 1000,
            'resizeTimeElapsedMs': resizeTimeElapsed * 1000,
            'resizeImagePath': imgOutputPath,
            'classifyOutput': output,
            'classifyTimeElapsedMs': classifyTimeElapsed * 1000
        }

api.add_resource(Classify, '/classify', endpoint = 'classify')
