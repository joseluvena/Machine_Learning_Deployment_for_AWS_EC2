import flask
from flask import Flask, render_template
from flask import request

app = Flask(__name__)

from flask_cors import CORS
CORS(app)

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/about')
def about():
	return render_template('about.html')

@app.route('/sources')
def sources():
	return render_template('sources.html')

@app.route('/predict', methods=["POST"])
def predict():
	gre = request.form.get('gre')
	toefl = request.form.get('toefl')
	unirating = request.form.get('unirating')
	sop = request.form.get('sop')
	lor = request.form.get('lor')
	cgpa = request.form.get('cgpa')
	research = request.form.get('research')

	import pickle
	from sklearn import linear_model
	with open('model_pickle', 'rb') as f:
		mod = pickle.load(f)
	
	import numpy as np
	
	inputs = (np.array([gre, 
					  toefl,
					  unirating,
					  sop,
					  lor,
					  cgpa,
					  research])).reshape(1,-1)
	#import sklearn
	admission_chance = str(100*(mod.predict(inputs)))
	
	return render_template('prediction.html', admission_chance=admission_chance)

app.run(debug=True)

#http://127.0.0.1:5000/predict?inputgre=340&inputtoefl=110&inputunirating=5&inputsop=5&inputlor=4&inputcgpa=9.1&inputresearch=0
