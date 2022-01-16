FROM tensorflow/tensorflow:latest-gpu-jupyter

ARG USERNAME=appuser
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Define working dir
ARG WORKDIR="/tmp/pip-tmp/"

# Copy file in working dir
COPY setup.py $WORKDIR

# Install packages
RUN pip install --upgrade pip setuptools wheel 
RUN python -m pip --disable-pip-version-check --no-cache-dir install -e $WORKDIR[dev]   \
    && rm -rf ${WORKDIR}


# [Optional] Set the default user. Omit if you want to keep the default as root.
USER $USERNAME