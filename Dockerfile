FROM tensorflow/tensorflow:latest-gpu-jupyter

# Define working dir
ARG WORKDIR="/tmp/pip-tmp/"

# Copy file in working dir
COPY setup.py $WORKDIR

# Install packages
RUN pip install --upgrade pip setuptools wheel 
RUN python -m pip --disable-pip-version-check --no-cache-dir install -e $WORKDIR[dev]   \
    && rm -rf ${WORKDIR}