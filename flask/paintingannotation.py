from flask import Flask, redirect, url_for, render_template, request, session, flash, Markup
from datetime import timedelta
import os
import json
import random

app = Flask(__name__)
app.secret_key = "sdf79w3hsdfz83250fdg1287sdgf546dfg"

html_files = []
htmlfiles_metadata = {}
for path, subdirs, files in os.walk('retrieval/data'):
	for name in files:
		html_path = os.path.join(path, name)
		if '.json' in html_path:
			continue
		html_files.append(html_path)
		with open(os.path.join(os.path.dirname(html_path), 'metadata', os.path.basename(html_path).replace('.html', '.json'))) as f:
			metadata = json.load(f)
			htmlfiles_metadata[html_path] = metadata

def getrandomfiles():
	print("Get random files")
	global html_files
	rhtml_files = random.sample(html_files, 20)
	hppaths = []
	sgpaths = []
	for filepath in rhtml_files:
		if filepath.find('scenegraphs') != -1:
			sgpaths.append(filepath)
		else:
			hppaths.append(filepath)
	random.shuffle(hppaths)
	random.shuffle(sgpaths)
	rhtmlfiles = hppaths + sgpaths
	return rhtmlfiles


@app.route("/artuserstudy/annotate", methods=['GET', 'POST'])
def annotate():
	global htmlfiles_metadata

	fileindex = session["fileindex"]
	rhtml_files = session['rhtmls']

	if "user" not in session: #not logged in
		return redirect(url_for("login"))

	if request.method == 'POST':
		print(request.form.getlist('image-checkbox'))
		print(fileindex)
		foldername = os.path.basename(os.path.dirname(rhtml_files[fileindex]))
		fileid = os.path.splitext(os.path.basename(rhtml_files[fileindex]))[0]
		filename = "%s_%s.json"%(fileid, session["user"])

		with open(os.path.join('results', foldername, filename), 'w') as f:
			annotations = request.form.getlist('image-checkbox')
			metadata = htmlfiles_metadata[rhtml_files[fileindex]]
			data = {'query': metadata['querypath'], 'ranking': metadata['resultpath'], 'annotations': annotations, 'retrievalfiles': metadata['retrievaltopk']}
			json.dump(data, f, indent=4, separators=(',', ': '))

		fileindex += 1
		session["fileindex"] = fileindex
		if fileindex >= len(rhtml_files):
			return redirect(url_for("logout"))
		if 'scenegraphs' in rhtml_files[fileindex] and session['section'] == 1:
			return redirect(url_for("scenegraph"))
			
		with open(rhtml_files[fileindex], 'r') as f:
			html_imagegrid = f.read()

		return render_template("base.html", content=html_imagegrid, pagenum=fileindex+1, pagenumtotal=len(rhtml_files), section=session["section"])
	else:
		with open(rhtml_files[fileindex], 'r') as f:
			html_imagegrid = f.read()
		return render_template("base.html", content=html_imagegrid, pagenum=fileindex+1, pagenumtotal=len(rhtml_files), section=session["section"])

@app.route("/artuserstudy/scenegraph-start", methods=["POST", "GET"])
def scenegraph():
	if request.method == "POST":
		session['section'] = 2
		return redirect(url_for("annotate"))
	else:
		return render_template("scenegraph-start.html")

@app.route("/", methods=["GET"])
def rootpage():
	return redirect(url_for("login"))

@app.route("/artuserstudy/login", methods=["POST", "GET"])
def login():
	if request.method == "POST":
		#session.permanent = True
		user = request.form["username"]
		if not user:
			return render_template("login.html")
		session["user"] = ''.join(user.split()).lower()
		print("User:", session["user"])
		print("Login:", session['rhtmls'])

		session["fileindex"] = 0
		session["section"] = 1 
		return redirect(url_for("instructions"))
	else:
		#if "user" in session:
		#	return redirect(url_for("annotate"))
		session['rhtmls'] = ['/home/althausc/master_thesis_impl/flask/retrieval/data/scenegraphs/2.html'] #getrandomfiles()

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
		message = Markup("Message send!")
		flash(message)
		return render_template("logout.html")
	else:
		return render_template("logout.html")

if __name__ == "__main__":
	app.run(debug=True)