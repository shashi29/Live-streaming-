install:
	pip install -r requirements.txt
	bash ffmpeg.sh

lint:
	pylint your_code_directory
	black your_code_directory

docker-build:
	docker build -t your_docker_image_name .

docker-test:
	docker run your_docker_image_name pytest

unit-test:
	pytest your_tests_directory

.PHONY: install lint docker-build docker-test unit-test
