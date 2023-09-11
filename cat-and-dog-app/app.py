from flask import Flask, render_template, request
import os

from keras.utils import load_img, img_to_array
from keras import models
import numpy as np

app = Flask(__name__)

IMG_FOLDER = os.path.join('static', 'IMG')

app.config['UPLOAD_FOLDER'] = IMG_FOLDER

@app.route('/', methods = ['GET'])
def hello_world():
    return render_template('index.html')

# @app.route("/")
# def Display_IMG():
#     Flask_Logo = os.path.join(app.config['UPLOAD_FOLDER'], 'flask-logo.png')
#     return render_template("temp_index.html", user_image=Flask_Logo)

@app.route('/', methods = ['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = './cat-and-dog-app/' + os.path.join(app.config['UPLOAD_FOLDER'], imagefile.filename)
    image_path_shortened = os.path.join(app.config['UPLOAD_FOLDER'], imagefile.filename)
    imagefile.save(image_path)

    image = load_img(image_path, target_size = (150,150))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    model = models.load_model('cats_and_dogs_small_1.h5')
    yhat = model.predict(image)
    label = np.where(yhat[0][0] == 1.0, 'Dog', 'Cat')
    classification = '{}'.format(label)

    return render_template('index.html', prediction = classification, user_image = image_path_shortened)

if __name__ == '__main__':
    app.run(port = 3000, debug = True)