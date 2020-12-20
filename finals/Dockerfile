# Set up the image, copy the essentials, run apt-get logic(slides), bash script.

FROM ubuntu
RUN mkdir /finals
#RUN chown 1000:1000 /Assign_Docker
# The above line is not required for general users, I was facing some system issue and getting
# permission denied error, hence used this. The below line will be enough for generally running the script.
RUN chmod a+x /finals

# Coping the baseball database
COPY ./baseball.sql/ /finals/baseball.sql

# Coping our bash file
COPY ./Finals_bash.sh/ /finals/Finals_bash.sh

# Coping the SQL script which gets us our features
COPY ./Finals.sql/ /finals/Finals.sql

# Coping other required files
COPY ./Assignment4_FeatureEngineering.py/ /finals/Assignment4_FeatureEngineering.py
COPY ./Midterm_VaishnaviBihare.py/ /finals/Midterm_VaishnaviBihare.py
COPY ./requirements.txt .

RUN chmod +x /finals/Finals_bash.sh

RUN apt-get update \
   && apt-get install -y mysql-client\
      build-essential\
      python3\
      python\
      python3-pip\
      pyhton3-dev\
      python3-pymysql\
 && rm -rf/ var/lib/apt/lists/*

RUN pip install --compile --no-cache-dir -r requirements.txt
RUN chmod +x/ finals/ Finals_bash.sh


CMD ./finals/finals_vb_bash.sh
