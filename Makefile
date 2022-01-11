# Check if GPU is available
NVCC_RESULT := $(shell which nvcc 2> NULL)
NVCC_TEST := $(notdir $(NVCC_RESULT))
ifeq ($(NVCC_TEST),nvcc)
GPUS=--gpus all
else
GPUS=
endif

# For Windows use CURDIR
ifeq ($(PWD),)
PWD := $(CURDIR)
endif

# Set flag for docker run command
RUN_FLAGS=-it --rm  ${GPUS} -v ${PWD}:/home/app/code-acme -w /home/app/code-acme 

# Default version is tf-core
version = 0.0
DOCKER_IMAGE_NAME = code-acme
DOCKER_IMAGE_TAG = $(version)
DOCKER_RUN=docker run $(RUN_FLAGS) $(IMAGE)

IMAGE = $(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_TAG)
# make file commands
build:
	DOCKER_BUILDKIT=1 docker build --tag $(IMAGE) .

bash:
	$(DOCKER_RUN) bash

#run-tests:
#	$(DOCKER_RUN) /bin/bash bash_scripts/tests.sh

#run:
#	$(DOCKER_RUN) python $(example) --base_dir /home/app/mava/logs/	