from flask import Flask, redirect, url_for, render_template, request, session
from datetime import timedelta
import os
import json
import random

app = Flask(__name__)
app.secret_key = "sdf79w3hsdfz83250fdg1287sdgf546dfg"

rhtml_files = []
for path, subdirs, files in os.walk('retrieval'):
	for name in files:
		rhtml_files.append(os.path.join(path, name))
rhtml_files.sort()
random.shuffle(rhtml_files)
print(rhtml_files)
fileindex = 0

@app.route("/artuserstudy/annotate", methods=['GET', 'POST'])
def annotate():
	fileindex = session["fileindex"]
	
	if "user" not in session: #not logged in
		return redirect(url_for("login"))

	if request.method == 'POST':
		print(request.form.getlist('image-checkbox'))
		print(fileindex)
		foldername = os.path.basename(os.path.dirname(rhtml_files[fileindex]))
		filename = "%d_%s.json"%(fileindex, session["user"])

		with open(os.path.join('results', foldername, filename), 'w') as f:
			json.dump(request.form.getlist('image-checkbox'), f)

		fileindex += 1
		session["fileindex"] = fileindex
		if fileindex >= len(rhtml_files):
			return redirect(url_for("logout"))
		
		with open(rhtml_files[fileindex], 'r') as f:
			html_imagegrid = f.read()
		
		return render_template("base.html", content=html_imagegrid, pagenum=fileindex+1, pagenumtotal=len(rhtml_files))
	else:
		with open(rhtml_files[fileindex], 'r') as f:
			html_imagegrid = f.read()
		return render_template("base.html", content=html_imagegrid, pagenum=fileindex+1, pagenumtotal=len(rhtml_files))

@app.route("/artuserstudy/login", methods=["POST", "GET"])
def login():
	if request.method == "POST":
		#session.permanent = True
		user = request.form["username"]
		session["user"] = ''.join(user.split()).lower()
		print("USER:", session["user"])
		session["fileindex"] = 0
		return redirect(url_for("instructions"))
	else:
		#if "user" in session:
		#	return redirect(url_for("annotate"))

		return render_template("login.html")

@app.route("/artuserstudy/instructions", methods=["POST", "GET"])
def instructions():
	if request.method == "POST":
		return redirect(url_for("annotate"))
	else:
		if "user" not in session:
			return redirect(url_for("login"))

		return render_template("taskdescription.html")

@app.route("/artuserstudy/logout", methods=["POST", "GET"])
def logout():
	if request.method == "POST":
		feedback = request.form["feedback"]
		filename = '/home/althausc/master_thesis_impl/flask/results/feedback/%s.json'%(session["user"])
		with open(filename, 'w') as f:
			json.dump(feedback, f)
		return render_template("blank.html")
	else:
		return render_template("logout.html")

if __name__ == "__main__":
	app.run(debug=True)