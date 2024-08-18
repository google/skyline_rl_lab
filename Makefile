.PHONY: init test lint dist

all: init test lint

init:
	pip3 install -r requirements.txt

dist: init test
	rm -f dist/*
	python3 setup.py sdist bdist_wheel

lint:
	flake8 skyline

upload: dist
	twine upload --skip-existing dist/*
