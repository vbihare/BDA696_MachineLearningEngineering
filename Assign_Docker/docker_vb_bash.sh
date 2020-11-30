# Bash script to check if we have the Baseball data or no, if not create one.
# Will be used to call our SQL file that calculates the rolling average.

sleep 10

DATA=`mysqlshow -h assign-baseball-data -psecret -u root baseball`

# Checking if the database exists or not
if ! mysql -h assign-baseball-data -uroot -psecret -e 'use baseball'; then
  echo "Baseball DOES NOT exists"
  mysql -h assign-baseball-data -psecret -u root -e \
  "CREATE DATABASE IF NOT EXISTS baseball"
  mysql -h assign-baseball-data -psecret -u root baseball < /Assign_Docker/baseball.sql
else
  echo "Baseball DOES exists"
fi

# Calling the SQL file to calculate the rolling average
mysql -h assign-baseball-data -psecret -u root baseball < /Assign_Docker/docker_assign.sql

mysql -h assign-baseball-data -u root -psecret -e '
  USE baseball;
  SELECT * FROM rolling_avg;' > /results/results.txt
