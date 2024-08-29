FROM python:3.10-slim-bullseye
ARG UID=1000
ARG GID=1000
ARG TZ=Europe/Rome

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN echo 'vm.overcommit_memory=1' >> /etc/sysctl.conf

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        build-essential \
        curl \
        git \
        gcc \
        ffmpeg \
        locales \
        wget


RUN sed -i '/it_IT.UTF-8/s/^# //g' /etc/locale.gen && \
    locale-gen
ENV LANG it_IT.UTF-8  
ENV LANGUAGE it_IT:it  
ENV LC_ALL it_IT.UTF-8

RUN groupadd --gid $GID user
RUN useradd --no-log-init --create-home --shell /bin/bash --uid $UID --gid $GID user
USER user
ENV HOME=/home/user
WORKDIR $HOME
RUN mkdir $HOME/.cache $HOME/.config && chmod -R 777 $HOME
ENV PATH="$HOME/.local/bin:$PATH"
        
WORKDIR $HOME/xtts-rvc-flask-voice-clone
ENV PATH="/home/uwsgi/.local/bin:${PATH}"

COPY requirements.txt .

RUN pip3 install -r requirements.txt

USER root
ENV HOME=/home/user
ADD --chown=user:user hubert $HOME/xtts-rvc-flask-voice-clone/hubert
ADD --chown=user:user RVC $HOME/xtts-rvc-flask-voice-clone/RVC
ADD --chown=user:user nuwave2 $HOME/xtts-rvc-flask-voice-clone/nuwave2
ADD --chown=user:user utils $HOME/xtts-rvc-flask-voice-clone/utils
ADD --chown=user:user config.py $HOME/xtts-rvc-flask-voice-clone/config.py
ADD --chown=user:user main.py $HOME/xtts-rvc-flask-voice-clone/main.py
ADD --chown=user:user voice.py $HOME/xtts-rvc-flask-voice-clone/voice.py
ADD --chown=user:user get_model.py $HOME/xtts-rvc-flask-voice-clone/get_model.py
ADD --chown=user:user init.py $HOME/xtts-rvc-flask-voice-clone/init.py
ADD --chown=user:user entrypoint.sh $HOME/xtts-rvc-flask-voice-clone/entrypoint.sh
ADD --chown=user:user uwsgi.ini $HOME/xtts-rvc-flask-voice-clone/uwsgi.ini
RUN chmod +x $HOME/xtts-rvc-flask-voice-clone/entrypoint.sh

RUN chown -R user:user $HOME/xtts-rvc-flask-voice-clone
RUN chmod 777 -R $HOME/xtts-rvc-flask-voice-clone


USER user
CMD ["./entrypoint.sh"]