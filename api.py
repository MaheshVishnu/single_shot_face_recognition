import flask
import pygame
import pygame.camera
import os, shutil
import PIL
import sys
import cv2
import pickle
import numpy as np
import math
import face_recognition
from glob import glob
from scipy import misc
from flask import Flask, render_template, request, redirect, flash
from werkzeug.utils import secure_filename
from sklearn.externals import joblib
from sklearn import neighbors
from PIL import Image, ImageDraw
from face_recognition.face_recognition_cli import image_files_in_folder

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
camera = cv2.VideoCapture(0)
@app.route("/")
@app.route("/index")
def index():
   return flask.render_template('index.html')
@app.route('/predict', methods=['POST'])
def make_prediction():
	if request.method=='POST':
		path = '/home/sreemahavishnu/Desktop/programming/courses/udacity/deep_learning/facial_recognition/image1.jpg'
		img = face_recognition.load_image_file(path)
		#print(img)
		#img = img.reshape(1, -1)
		X_face_locations = face_recognition.face_locations(img)
		#print(X_face_locations)
		faces_encodings = face_recognition.face_encodings(img, known_face_locations=X_face_locations)
		#print(faces_encodings)
		closest_distances = model.kneighbors(faces_encodings, n_neighbors=1)
		are_matches = [closest_distances[0][i][0] <= 0.6 for i in range(len(X_face_locations))]
		prediction = [pred if rec else ("unknown", loc) for pred, loc, rec in zip(model.predict(faces_encodings), X_face_locations, are_matches)]
		label = str(prediction)
		return render_template('index.html', label=label)
@app.route('/snap', methods=['POST'])
def take_picture(port = 0):
	if request.method == 'POST' :
		camera = cv2.VideoCapture(0)
		print(camera.isOpened())
		#camera.set(3, 1280)
		#camera.set(4, 720)
		for i in range(0, 30):
			temp = camera.read()
		retval, im = camera.read()
		cv2.imwrite('image1.jpg',im)
		camera.release()
		del(camera)
		'''camera_port = 0
		ramp_frames = 30
		camera = cv2.VideoCapture(0)
		retval, im = camera.read()
		for i in range(0, ramp_frames):
			temp = camera.read()
		filename = "aj.jpg"
		cv2.imwrite(filename,im)
		del(camera)'''
		return render_template('index.html', label = "picture captured")
@app.route('/predict_capture_pygame', methods = ['POST'])
def predict_capture_pygame() :
	if request.method == 'POST' :
		pygame.camera.init()
		camera = pygame.camera.Camera('/dev/video0', (800, 600))
		camera.start()
		for i in range(0, 30) :
			temp = camera.get_image()
		img = camera.get_image()
		pygame.image.save(img, 'image1.jpg')
		camera.stop()
@app.route('/train_snap', methods = ['POST'])
def train_capture() :
	if request.method == 'POST' :
		camera = cv2.VideoCapture(0)
		print(camera.isOpened())
		#camera.set(3, 1280)
		#camera.set(4, 720)
		for i in range(0, 30):
			temp = camera.read()
		retval, im = camera.read()
		cv2.imwrite('photos_temp/image1.jpg',im)
		camera.release()
		del(camera)
		'''camera_port = -1
		ramp_frames = 30
		camera = cv2.VideoCapture(0)
		retval, im = camera.read()
		for i in range(0, ramp_frames):
			temp = camera.read()
		filename = "aj.jpg"
		cv2.imwrite(filename,im)
		del(camera)'''
		return render_template('index.html', label = "picture captured")
@app.route('/train_capture_pygame', methods = ['POST'])
def train_capture_pygame() :
	if request.method == 'POST' :
		pygame.camera.init()
		camera = pygame.camera.Camera('/dev/video0', (800, 600))
		camera.start()
		for i in range(0, 30) :
			temp = camera.get_image()
		img = camera.get_image()
		pygame.image.save(img, 'photos_temp/image1.jpg')
		camera.stop()
@app.route('/train', methods = ['POST'])
def train_image() :
	if request.method == 'POST' :
		text = request.form['name']
		train_path = '/home/sreemahavishnu/Desktop/programming/courses/udacity/deep_learning/facial_recognition/10k_small/dynamic_train/'
		src = '/home/sreemahavishnu/Desktop/programming/courses/udacity/deep_learning/facial_recognition/photos_temp/image1.jpg'
		os.makedirs(train_path+text)
		shutil.copy2(src, train_path+text+'/'+'image1.jpg')
		classifier = train(train_path, model_save_path="trained_knn_model.clf", n_neighbors=2)
		joblib.dump(classifier, 'model.pkl')
		return render_template('index.html', label = "trained")
def train(train_dir='/home/sreemahavishnu/Desktop/programming/courses/udacity/deep_learning/facial_recognition/10k_small', model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    X = []
    y = []
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            print("doing1")
            continue
        #print("doing")
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) != 1:
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)
    #print(X.shape())
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)
    return knn_clf
if __name__ == '__main__':
	app.secret_key = 'key'
	app.config['SESSION_TYPE'] = 'filesystem'
	model = joblib.load('model.pkl')
	app.run(host='0.0.0.0', port=8000, debug=True)
