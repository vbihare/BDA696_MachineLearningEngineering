# Bash script to check if we have the Baseball data or no, if not create one.
# Will be used to call our SQL file that calculates the rolling average.

DATA=`mysqlshow -h assign_baseball_data -u root baseball`

# Checking if the database exists or not
if ["$DATA" == "baseball"] ; then
  echo "Baseball data exists"

# If the database doesn't exists then we will create the Baseball database
else
  mysql -h assign_baseball_data -u root -e \
  "CREATE DATABASE IF NOT EXISTS baseball"
  mysql -h assign_baseball_data -u root baseball < /Assign_Docker/baseball.sql
fi

# Calling the SQL file to calculate the rolling average
mysql -h assign_baseball_data -u root baseball < /Assign_Docker/docker_assign.sql