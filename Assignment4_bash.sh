pip-compile --output-file=requirements.txt requirements.in --upgrade

pip3 install -r requirements.txt

pre-commit install

pre-commit run --all-files

read -p "please enter csv name " file
read -p "please enter the response variable name" response

echo "Calling Assignment 4 Python Code"
python3 Assignment4_FeatureEnginerring.py $file $response