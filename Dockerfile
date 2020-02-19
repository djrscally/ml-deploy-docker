FROM ubuntu:18.04

# Install dependencies
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y python3 python3-pip apache2

# Create the application directories
RUN mkdir -p /var/ModelEndpoint
RUN mkdir -p /var/ModelEndpoint/artifacts

# Copy the pickled model and files
ADD artifacts/model.pkl /var/ModelEndpoint/artifacts/
ADD app.py /var/ModelEndpoint/
ADD requirements.txt /var/ModelEndpoint/
ADD servemodel.sh /var/ModelEndpoint/

# We need to make the .sh file executable
RUN chmod +x /var/ModelEndpoint/servemodel.sh

# Install all the python dependencies
RUN python3 -m pip install -r /var/ModelEndpoint/requirements.txt

# Copy over the apache conf file and perform the required config
ADD model.conf /etc/apache2/sites-available/
RUN a2enmod proxy_http
RUN a2ensite model
RUN echo "Listen 5000" >> /etc/apache2/ports.conf

WORKDIR /var/ModelEndpoint/

# Run the script to start apache and gunicorn
CMD /var/ModelEndpoint/servemodel.sh
