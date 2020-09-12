pip-compile --output-file=requirements.txt requirements.in --upgrade

pip3 install -r requirements.txt

pre-commit install

pre-commit run --all-files

echo "Calling the BDA696 Assignment 1 python Code"
# Run the Assignment file
python ./BDA696_Assignment1.py