setup:
	rm -rf venv
	python3.7 -m venv venv

init:
	pip install --upgrade pip
	pip install -r requirements.txt
