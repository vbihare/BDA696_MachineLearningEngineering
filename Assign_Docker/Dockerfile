FROM ubuntu

RUN mkdir /Assign_Docker
RUN chown 1000:1000 /Assign_Docker
RUN chmod a+x /Assign_Docker

# Coping the baseball database
COPY ./baseball.sql/ Assign_Docker/baseball.sql

# Coping our bash file
COPY ./docker_vb_bash.sh/ Assign_Docker/docker_vb_bash.sh

# Coping the SQL script which gets us the Rolling average
COPY ./docker_assign.sql/ Assign_Docker/docker_assign.sql

RUN apt-get update \
  && apt-get install --no-install-recommends --yes \
     mysql-client \
     wget \
  && rm -rf /var/lib/apt/lists/*

RUN chmod +x /Assign_Docker/docker_vb_bash.sh
CMD ./Assign_Docker/docker_vb_bash.sh


