#!/usr/bin/env bash
python init.py
uwsgi --ini uwsgi.ini --enable-threads --py-call-uwsgi-fork-hooks --log-4xx --log-5xx --lazy
#uwsgi --ini /app/uwsgi.ini --enable-threads
