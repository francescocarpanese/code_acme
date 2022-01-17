# For Windows use CURDIR
ifeq ($(PWD),)
PWD := $(CURDIR)
endif

# Set flag for docker run command
RUN_FLAGS= --rm 
MOUNT_FLAGS= -v  ${PWD}:/home/app/code-acme -w /home/app/code-acme
INTERACTIVE_FLAGS= -it -p 8888:8888
GPUS= --gpus all 

version = 0.0
DOCKER_IMAGE_NAME = code-acme
DOCKER_IMAGE_TAG = $(version)
IMAGE = $(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_TAG)

build:
	DOCKER_BUILDKIT=1 docker build --tag $(IMAGE) .

bash:
	docker run $(RUN_FLAGS) ${INTERACTIVE_FLAGS} ${MOUNT_FLAGS} $(IMAGE) bash

bash-gpu:
	docker run $(RUN_FLAGS) ${INTERACTIVE_FLAGS} ${GPUS} ${MOUNT_FLAGS} $(IMAGE) bash

# Non interactive shell for github actions
test: 
	docker run  $(RUN_FLAGS) ${MOUNT_FLAGS} $(IMAGE) /bin/bash -c "pip install .; pytest"