FROM trestletech/plumber
MAINTAINER Carlos Catania (AKA Harpo) "harpomaxx@gmail.com"
RUN apt install -y python-pip virtualenv
RUN R -e "install.packages('keras')"
RUN R -e "library(keras);install_keras()"
COPY . /app
WORKDIR /app
CMD ["/app/launchservice.R"]

