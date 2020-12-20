# Bash script to check if we have the Baseball data or no, if not create one.
# Will be used to call our SQL file that calculates the rolling average.

sleep 10

# DATA=`mysqlshow -h assign-baseball-data -psecret -u root baseball`

# Checking if the database exists or not
if ! mysql -h final-baseball-data -uroot -e 'use baseball'; then
  echo "Baseball DOES NOT exists"
  mysql -h true-baseball-data -u root -e \
  "CREATE DATABASE IF NOT EXISTS baseball"
  mysql -h final-baseball-data -u root baseball < /final/baseball.sql
else
  echo "Baseball DOES exists"
fi

# Calling the SQL file to calculate the rolling average
mysql -h final-baseball-data -psecret -u root baseball < /final/Finals.sql

mysql -h assign-baseball-data -u root -e '
  USE baseball;
  SELECT * FROM features;' > /final/feature.csv

# Calling the python script
python Assignment4_FeatureEngineering.py
python Midterm_VaishnaviBihare.py
python Finals_Vaishnavi.py
