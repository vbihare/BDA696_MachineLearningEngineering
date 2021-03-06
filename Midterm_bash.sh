pip-compile --output-file=requirements.txt requirements.in --upgrade

pip3 install -r requirements.txt

pre-commit install

pre-commit run --all-files

read -p "please enter csv name " file
read -p "please enter the response variable name" response
if [[ $response == '' ]] && [[ $file != '' ]];then
  read -p "Enter a response variable:" response
  if [ -z $target ];then
     '$file' = 'auto-mpg.csv'
     '$response' = 'mpg'
  fi
fi

echo $file

echo "Calling Assignment 4 Python Code"
python3 Midterm_VaishnaviBihare.py $file $response