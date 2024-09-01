import os
import sys
import voice
import logging
import voice
import uuid
import utils
import json
import hashlib
import shutil
import threading
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask import Flask, request, send_file, Response, jsonify, make_response, after_this_request, g
from flask_restx import Api, Resource, reqparse
from flask_caching import Cache
from os.path import join, dirname
from dotenv import load_dotenv
from time import strftime
from pathlib import Path


import torch.multiprocessing as mp

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

logging.basicConfig(
        handlers=[
            logging.FileHandler(os.path.dirname(os.path.abspath(__file__)) + "/logs/uwsgi.log", mode="w"),
            logging.StreamHandler()
        ],
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=int(os.environ.get("LOG_LEVEL")),
        datefmt='%Y-%m-%d %H:%M:%S')

log = logging.getLogger('werkzeug')
log.setLevel(int(os.environ.get("LOG_LEVEL")))


if not os.path.exists(os.environ.get("TMP_DIR")):
  os.mkdir(os.environ.get("TMP_DIR"))
mp.set_start_method('spawn', force=True)

app = Flask(__name__)
class Config:    
    CACHE_TYPE = os.environ['CACHE_TYPE']
    CACHE_REDIS_HOST = os.environ['CACHE_REDIS_HOST']
    CACHE_REDIS_PORT = os.environ['CACHE_REDIS_PORT']
    CACHE_REDIS_DB = os.environ['CACHE_REDIS_DB']
    CACHE_REDIS_URL = os.environ['CACHE_REDIS_URL']
    CACHE_DEFAULT_TIMEOUT = os.environ['CACHE_DEFAULT_TIMEOUT']
    SCHEDULER_API_ENABLED = False

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["120/minute"],
    storage_uri="memory://",
)

app.config.from_object(Config())

    
@app.after_request
def after_request(response):
  if not request.path.startswith('/utils/healthcheck'):
    timestamp = strftime('[%Y-%b-%d %H:%M]')
    logging.info('%s %s %s %s %s %s', timestamp, request.remote_addr, request.method, request.scheme, request.full_path, response.status)
  return response

cache = Cache(app)
api = Api(app)

def get_response_str(text: str, status):
  r = Response(response=text, status=status, mimetype="text/xml")
  r.headers["Content-Type"] = "text/xml; charset=utf-8"
  return r

def get_response_json(data, status):
  r = Response(response=data, status=status, mimetype="application/json")
  r.headers["Content-Type"] = "application/json; charset=utf-8"
  r.headers["Accept"] = "application/json"
  return r

nsvoice = api.namespace('voice', 'Voice APIs')

@nsvoice.route('/clone/<string:voice_name>/')
@nsvoice.route('/clone/<string:voice_name>/<int:restart>/')
@nsvoice.route('/clone/<string:voice_name>/<int:restart>/<int:epochs>/')
class CloneClass(Resource):
  def post (self, voice_name: str, restart = 0, epochs = 100):
    try:      
      found = False
      for thread in threading.enumerate(): 
        if thread.name.startswith('voice_clone_') and thread.is_alive():
          found = True
      if found:    
        data = {
          "message": "A voice clone process is running, please wait for it to finish before calling the clone API again!"
        } 
        return get_response_json(json.dumps(data), 206)
      else:
        i = 1
        datasets = []
        traindir_path =  os.path.dirname(os.path.abspath(__file__)) + "/datasets/" + voice_name
        while request.files.get('dataset_' + str(i), None) is not None:
          datasets.append(request.files.get('dataset_'+ str(i), None))
          i = i + 1
        if len(datasets) == 0 and (not os.path.isdir(traindir_path) or (os.path.isdir(traindir_path) and len(os.listdir(traindir_path)) == 0)):
              return make_response('Dataset folder is empty and no dataset provided! You need to provide at least one dataset_x in wav format, starting from 1 (es. dataset_1, dataset_2, etc...)', 400)
        else:
          for dataset in datasets:
            if not dataset.filename.endswith(".wav"):
              return make_response('Every "dataset_x" param extension must be .wav or .mp3!', 400)

          if len(datasets) > 0 and os.path.exists(traindir_path): 
            shutil.rmtree(traindir_path)
          if not os.path.exists(traindir_path): 
            os.makedirs(traindir_path)

          dataset_path = None
          dataset_paths = []
          i = 1
          for dataset in datasets:
            dataset_path = traindir_path + "/dataset_" + str(i) + ".wav"
            dataset.save(dataset_path)
            dataset_paths.append(dataset_path)
            i = i + 1
          job_id = uuid.uuid4().hex
          voice_clone_thread = "voice_clone_" + voice_name + "_" + job_id
          threading.Thread(target=lambda: voice.voice_clone(voice_name, job_id, epochs, dataset_paths, restart==1), name=voice_clone_thread).start()
          data = {
            "message": "Starting voice cloning process",
            "epochs": str(epochs),
            "voice_name": voice_name,
            "job_id": job_id,
            "status_url": request.root_url + "voice/clone_status/" + voice_name + "/" + job_id
          }
          return get_response_json(json.dumps(data), 200)  
    except Exception as e:
      exc_type, exc_obj, exc_tb = sys.exc_info()
      fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
      logging.error("%s %s %s", exc_type, fname, exc_tb.tb_lineno, exc_info=1)
      g.request_error = str(e)

@nsvoice.route('/clone_status/<string:voice_name>/<string:job_id>')
class CloneStatusClass(Resource):
  def get (self, voice_name: str, job_id: str):
    try:
      voice_clone_thread = "voice_clone_" + voice_name + "_" + job_id
      found = False
      for thread in threading.enumerate(): 
        if thread.name == voice_clone_thread and thread.is_alive():
          found = True
      data = {
        "status": "Running" if found else "Inactive",
        "voice_name": voice_name,
        "job_id": job_id,
      }
      log = {}
      count = 0
      if os.path.isfile(os.path.dirname(os.path.abspath(__file__)) + "/RVC/logs/" + voice_name + "/train.log"):
        for line in reversed(list(open(os.path.dirname(os.path.abspath(__file__)) + "/RVC/logs/" + voice_name + "/train.log"))):
          log[str(count)] = line.replace("\n","").replace("\t",":")
          count = count + 1
      if os.path.isfile(os.path.dirname(os.path.abspath(__file__)) + "/RVC/logs/" + voice_name + "/extract_f0_feature.log"):
        for line in reversed(list(open(os.path.dirname(os.path.abspath(__file__)) + "/RVC/logs/" + voice_name + "/extract_f0_feature.log"))):
          log[str(count)] = line.replace("\n","").replace("\t",":")
          count = count + 1
      if os.path.isfile(os.path.dirname(os.path.abspath(__file__)) + "/RVC/logs/" + voice_name + "/preprocess.log"):
        for line in reversed(list(open(os.path.dirname(os.path.abspath(__file__)) + "/RVC/logs/" + voice_name + "/preprocess.log"))):
          log[str(count)] = line.replace("\n","").replace("\t",":")
          count = count + 1
      if os.path.isfile(os.path.dirname(os.path.abspath(__file__)) + "/logs/uwsgi.log"):
        for line in reversed(list(open(os.path.dirname(os.path.abspath(__file__)) + "/logs/uwsgi.log"))):
          if job_id in line and "GET" not in line and "POST" not in line:
            log[str(count)] = line.replace("\n","").replace("\t",":").replace("["+job_id+"]","").strip()
            count = count + 1

      data["log"] = log
      return get_response_json(json.dumps(data), 200)   
    except Exception as e:
      exc_type, exc_obj, exc_tb = sys.exc_info()
      fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
      logging.error("%s %s %s", exc_type, fname, exc_tb.tb_lineno, exc_info=1)
      g.request_error = str(e)

@nsvoice.route('/talk/<string:voice_name>/<string:text>/')
@nsvoice.route('/talk/<string:voice_name>/<string:text>/<int:use_bark>/')
@nsvoice.route('/talk/<string:voice_name>/<string:text>/<int:use_bark>/<string:language>/')
class TalkClass(Resource):
  def get (self, voice_name: str, text: str, use_bark = 0, language = "it"):
    try:
      found = False
      for thread in threading.enumerate(): 
        if thread.name.startswith('voice_talk_') and thread.is_alive():
          found = True
      if found:    
        data = {
          "message": "An audio generation process is running, please wait for it to finish before calling the talk API again!"
        } 
        return get_response_json(json.dumps(data), 206)
      else:
        if os.path.isdir(os.path.dirname(os.path.abspath(__file__)) + "/RVC/logs/"+voice_name) and os.path.isfile(os.path.dirname(os.path.abspath(__file__)) + "/RVC/weights/"+voice_name+".pth") and os.path.isfile(os.path.dirname(os.path.abspath(__file__)) + "/bark/assets/prompts/"+voice_name+".npz"): 
          job_id = voice_name + "_" + uuid.uuid4().hex
          voice_talk_thread = "voice_talk_" + job_id
          threading.Thread(target=lambda: voice.talk(voice_name, text, job_id, language, use_bark==1), name=voice_talk_thread).start()
          data = {
            "message": "Starting audio generation process.",
            "voice_name": voice_name,
            "text": text,
            "job_id": job_id,
            "poll_url": request.root_url + "voice/poll/" + job_id
          }
          return get_response_json(json.dumps(data), 200)
        else:
          data = {
            "message": "voice_name not found on server."
          }
          return get_response_json(json.dumps(data), 404)

    except Exception as e:
      exc_type, exc_obj, exc_tb = sys.exc_info()
      fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
      logging.error("%s %s %s", exc_type, fname, exc_tb.tb_lineno, exc_info=1)
      g.request_error = str(e)

@nsvoice.route('/poll/<string:job_id>')
class PollClass(Resource):
  def get (self, job_id: str):
    try:
      voice_talk_thread = "voice_talk_" + job_id
      found = False
      for thread in threading.enumerate(): 
        if thread.name == voice_talk_thread and thread.is_alive():
          found = True
      if found:
        data = {
          "status": "Running",
          "job_id": job_id
        }
        return get_response_json(json.dumps(data), 206)    
      else:
        path = os.environ.get("TMP_DIR") + "/" + job_id + ".wav"
        if os.path.isfile(path):
          data = {
            "status": "Completed",
            "job_id": job_id,
            "audio_url": request.root_url + "voice/request_audio/" + job_id
          }
          return get_response_json(json.dumps(data), 200)    
        else:
          return make_response("job_id not found!", 500)    
    except Exception as e:
      exc_type, exc_obj, exc_tb = sys.exc_info()
      fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
      logging.error("%s %s %s", exc_type, fname, exc_tb.tb_lineno, exc_info=1)
      g.request_error = str(e)

@nsvoice.route('/request_audio/<string:job_id>')
class RequestFileClass(Resource):
  #@cache.cached(timeout=300, query_string=True)
  def get (self, job_id: str):
    try:
      filename =  job_id + ".wav"
      path = os.environ.get("TMP_DIR") + "/" + filename
      if os.path.isfile(path):
        out = voice.request_audio(path)
        if out is not None:
          return send_file(out, attachment_filename=filename, mimetype='audio/mpeg')
        else:
          return make_response("Audio generation error!", 500)
      else:
        return make_response("Audio not found!", 400)
    except Exception as e:
      exc_type, exc_obj, exc_tb = sys.exc_info()
      fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
      logging.error("%s %s %s", exc_type, fname, exc_tb.tb_lineno, exc_info=1)
      g.request_error = str(e)
      @after_this_request
      def clear_cache(response):
        cache.delete_memoized(TalkClass.get, self, str)
        return make_response(g.get('request_error'), 500)

@nsvoice.route('/listvoices')
class ListVoicesClass(Resource):
  #@cache.cached(timeout=60)
  def get(self):
    try:
      available_voices = voice.get_available_voices()
      if len(available_voices) == 0:
        data = {
          "message": "No voices available, please clone some voice using the clone API!"
        } 
        return get_response_json(json.dumps(data), 206)
      else:
        return get_response_json(json.dumps(available_voices), 200)
    except Exception as e:
      exc_type, exc_obj, exc_tb = sys.exc_info()
      fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
      logging.error("%s %s %s", exc_type, fname, exc_tb.tb_lineno, exc_info=1)
      cache.delete_memoized(ListVoicesClass.get, self, str)
      return make_response(str(e), 500)


nsutils = api.namespace('utils', 'Utils APIs')

@nsutils.route('/healthcheck')
class Healthcheck(Resource):
  def get (self):
    return "Ok!"