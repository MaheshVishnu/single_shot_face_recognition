import flask
import os
import numpy as np
import face_recognition
from scipy import misc
from flask import Flask, render_template, request, redirect, flash
from werkzeug.utils import secure_filename
from sklearn.externals import joblib

app = Flask(__name__)

@app.route("/")
@app.route("/index")
def index():
   return flask.render_template('index.html')

@app.route('/predict', methods=['POST'])
def make_prediction():
    if request.method=='POST':
    	if 'image' not in request.files:
    		#flash('No file found')
    		return redirect(request.url)
    	file = request.files['image']
    	if not file:
    		return render_template('index.html', label="No file")
    	filename = secure_filename(file.filename)
    	path = os.path.abspath(filename)
    	img = face_recognition.load_image_file(path)
    	flash(img.shape)
    	#img = img.reshape(1, 1)
    	X_face_locations = face_recognition.face_locations(img)
    	faces_encodings = face_recognition.face_encodings(img, known_face_locations=X_face_locations)
    	closest_distances = model.kneighbors(faces_encodings, n_neighbors=1)
    	are_matches = [closest_distances[0][i][0] <= 0.6 for i in range(len(X_face_locations))]
    	prediction = [pred if rec else ("unknown", loc) for pred, loc, rec in zip(model.predict(faces_encodings), X_face_locations, are_matches)]
    	label = str(prediction)
    	if label=='10':
    		label='0'
    	return render_template('index.html', label=label)
if __name__ == '__main__':
	app.secret_key = 'key'
	app.config['SESSION_TYPE'] = 'filesystem'
	model = joblib.load('model.pkl')
	app.run(host='0.0.0.0', port=8000, debug=True)