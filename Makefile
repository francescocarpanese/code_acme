# For Windows use CURDIR
ifeq ($(PWD),)
PWD := $(CURDIR)
endif

# Set flag for docker run command
RUN_FLAGS=-it --rm  --gpus all -v ${PWD}:/home/app/code-acme -w /home/app/code-acme -p 8888:8888

version = 0.0
DOCKER_IMAGE_NAME = code-acme
DOCKER_IMAGE_TAG = $(version)
DOCKER_RUN=docker run $(RUN_FLAGS) $(IMAGE)

IMAGE = $(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_TAG)

build:
	DOCKER_BUILDKIT=1 docker build --tag $(IMAGE) .

bash:
	$(DOCKER_RUN) bash