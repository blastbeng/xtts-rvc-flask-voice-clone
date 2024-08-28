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
        
WORKDIR $HOME/bark-flask-voice-clone
ENV PATH="/home/uwsgi/.local/bin:${PATH}"

COPY requirements.txt .

RUN pip3 install -r requirements.txt

USER root
ENV HOME=/home/user
RUN mkdir $HOME/bark-flask-voice-clone/bark
ADD --chown=user:user bark/__init__.py $HOME/bark-flask-voice-clone/bark/__init__.py
ADD --chown=user:user bark/api.py $HOME/bark-flask-voice-clone/bark/api.py
ADD --chown=user:user bark/generation.py $HOME/bark-flask-voice-clone/bark/generation.py
ADD --chown=user:user bark/model_fine.py $HOME/bark-flask-voice-clone/bark/model_fine.py
ADD --chown=user:user bark/model.py $HOME/bark-flask-voice-clone/bark/model.py
ADD --chown=user:user hubert $HOME/bark-flask-voice-clone/hubert
ADD --chown=user:user RVC $HOME/bark-flask-voice-clone/RVC
ADD --chown=user:user nuwave2 $HOME/bark-flask-voice-clone/nuwave2
ADD --chown=user:user utils $HOME/bark-flask-voice-clone/utils
ADD --chown=user:user config.py $HOME/bark-flask-voice-clone/config.py
ADD --chown=user:user main.py $HOME/bark-flask-voice-clone/main.py
ADD --chown=user:user voice.py $HOME/bark-flask-voice-clone/voice.py
ADD --chown=user:user get_model.py $HOME/bark-flask-voice-clone/get_model.py
ADD --chown=user:user init.py $HOME/bark-flask-voice-clone/init.py
ADD --chown=user:user entrypoint.sh $HOME/bark-flask-voice-clone/entrypoint.sh
ADD --chown=user:user uwsgi.ini $HOME/bark-flask-voice-clone/uwsgi.ini
RUN mkdir -p $HOME/bark/assets/prompts/.cache/huggingface/download/
RUN mkdir -p $HOME/bark-flask-voice-clone/bark/assets/prompts/.cache/huggingface/download/
RUN chmod +x $HOME/bark-flask-voice-clone/entrypoint.sh

RUN chown -R user:user $HOME/bark-flask-voice-clone
RUN chown -R user:user $HOME/bark
RUN chmod 777 -R $HOME/bark-flask-voice-clone
RUN chmod 777 -R $HOME/bark


USER user
CMD ["./entrypoint.sh"]