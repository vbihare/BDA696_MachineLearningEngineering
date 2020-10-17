#Run the following script to execute the Homework Assignment
pip-compile --output-file=requirements.txt requirements.in --upgrade

pip3 install -r requirements.txt

pre-commit install

pre-commit run --all-files

echo "Calling the BDA696 Assignment 3 python Code"
# Run the Assignment file
python ./Assignment3_Spark.py