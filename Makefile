build:
	pip install .

clean:
	pip uninstall greatdeluge -y
	rm -rf build/
	rm -rf dist
	rm -rf *.egg-info
	rm -rf GreatDeluge*/
	rm -rf */__pycache__

run:
	python greatDeluge.py