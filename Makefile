build:
	pip install .

build-docker:
	sudo docker build -f docker/Dockerfile --tag greatdeluge .

clean:
	pip uninstall greatdeluge -y
	rm -rf build/
	rm -rf dist
	rm -rf *.egg-info
	rm -rf GreatDeluge*/
	rm -rf */__pycache__

clean-docker:
	sudo docker system prune -af

run:
	python greatDeluge.py

run-docker:
	sudo docker run -e PYTHONUNBUFFERED=1 greatdeluge

extract-docker-data:
	container_id=$(docker create "$image")
	sudo docker cp "$(container_id):$(source_path)" "$(destination_path)"
	sudo docker rm "$container_id"