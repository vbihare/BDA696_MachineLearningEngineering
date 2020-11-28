DATA=`mysqlshow -h assign_baseball -u root baseball | grep -v Wildcard | grep -o baseball`

# Checking if the database exists or not
if ["$DATA" == "baseball"] ; then
  echo "Baseball data exists"

# If the database doesn't exists then we will create the Baseball database
else
  mysql -h assign_baseball -u root -e "CREATE DATABASE IF NOT EXISTS baseball"
  mysql -h assign_baseball -u root baseball < /Assign_Docker/baseball.sql

fi

# Calling the SQL file to calculate the rolling average
mysql -h assign_baseball -u root baseball < /Assign_Docker/docker_assign.sql