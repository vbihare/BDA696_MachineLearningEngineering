# Set up the image, copy the essentials, run apt-get logic(slides), bash script.

FROM ubuntu
RUN mkdir /Assign_Docker
#RUN chown 1000:1000 /Assign_Docker
# The above line is not required for general users, I was facing some system issue and getting
# permission denied error, hence used this. The below line will be enough for generally running the script.
RUN chmod a+x /Assign_Docker

# Coping the baseball database
COPY ./baseball.sql/ /Assign_Docker/baseball.sql

# Coping our bash file
COPY ./docker_vb_bash.sh/ /Assign_Docker/docker_vb_bash.sh

# Coping the SQL script which gets us the Rolling average
COPY ./docker_assign.sql/ /Assign_Docker/docker_assign.sql

RUN chmod +x /Assign_Docker/docker_vb_bash.sh

RUN apt-get update
RUN apt-get install -y mysql-client

CMD /Assign_Docker/docker_vb_bash.sh
