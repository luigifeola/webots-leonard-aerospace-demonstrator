ARG BASE_IMAGE=nvidia/cuda:12.6.0-base-ubuntu24.04
FROM ${BASE_IMAGE} AS downloader

# Determine Webots version to be used and set default argument
ARG WEBOTS_VERSION=R2023b
ARG WEBOTS_PACKAGE_PREFIX=

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install --yes wget bzip2 && rm -rf /var/lib/apt/lists/ && \
 wget https://github.com/cyberbotics/webots/releases/download/$WEBOTS_VERSION/webots-$WEBOTS_VERSION-x86-64$WEBOTS_PACKAGE_PREFIX.tar.bz2 && \
 tar xjf webots-*.tar.bz2 && rm webots-*.tar.bz2

FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

# UV
COPY --from=ghcr.io/astral-sh/uv:0.7.0 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 UV_SYSTEM_PYTHON=1

# Install Webots runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \ 
  build-essential \
  wget \
  xvfb \
  locales \
  python3 \
  python3-argcomplete \
  python-dev-is-python3 \
  python3-pip && \
  rm /usr/lib/python*/EXTERNALLY-MANAGED && \
  wget https://raw.githubusercontent.com/cyberbotics/webots/master/scripts/install/linux_runtime_dependencies.sh && \
  chmod +x linux_runtime_dependencies.sh && ./linux_runtime_dependencies.sh && rm ./linux_runtime_dependencies.sh && rm -rf /var/lib/apt/lists/

# Install Webots
WORKDIR /usr/local
COPY --from=downloader /webots /usr/local/webots/
ENV QTWEBENGINE_DISABLE_SANDBOX=1
ENV WEBOTS_HOME=/usr/local/webots
ENV PATH=/usr/local/webots:${PATH}

# Enable OpenGL capabilities
ENV NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility

# Set a user name to fix a warning
ENV USER=root

# Set the locales
RUN locale-gen en_US.UTF-8
ENV LANG='en_US.UTF-8' LANGUAGE='en_US:en' LC_ALL='en_US.UTF-8'

# Install python dependencies
WORKDIR /var/local
COPY requirements_dev.txt /var/local/requirements_dev.txt
COPY requirements.txt /var/local/requirements.txt
COPY services/tristan-yolo-py-inference/requirements.txt /var/local/services/requirements.txt
RUN uv pip install -r requirements_dev.txt -r requirements.txt -r services/requirements.txt

ARG USERNAME=user
ARG UID=1000
ARG GID=1000

# Rename the user
RUN usermod -l ${USERNAME} -d /home/${USERNAME} -m ubuntu

# Create workspace
WORKDIR /workspace
RUN chown -R ${UID}:${GID} /home/${USERNAME} &&\
        chown -R ${UID}:${GID} /workspace

# Ensure XDG dirs exist and are writable before volume mounts.
RUN mkdir -p /home/${USERNAME}/.config/Cyberbotics \
  /home/${USERNAME}/.cache/Cyberbotics \
  /home/${USERNAME}/.local/share/Cyberbotics \
  /home/${USERNAME}/.local/share/applications && \
    chown -R ${UID}:${GID} /home/${USERNAME}/.config \
  /home/${USERNAME}/.cache \
  /home/${USERNAME}/.local

USER $USERNAME