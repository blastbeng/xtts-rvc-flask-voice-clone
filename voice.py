import os
import string
import random
import uuid
import shutil
from pydub import AudioSegment
from encodec.utils import convert_audio
import numpy as np

from IPython.display import Audio


from TTS.api import TTS

from silenceremove import process as remove_silence

import torchaudio
import torch
from io import BytesIO
import psutil
from scipy.io import wavfile
import noisereduce as nr

import logging

from os.path import join, dirname
from dotenv import load_dotenv
dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)


logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=int(os.environ.get("LOG_LEVEL")),
        datefmt='%Y-%m-%d %H:%M:%S')

tts = False

def voice_clone(voice_name, epochs=100, cleanup=False, from_scratch=False):

  if from_scratch:
    if os.path.isfile(os.path.dirname(os.path.abspath(__file__)) + "/RVC/weights/"+voice_name+".pth"):
      os.remove(os.path.dirname(os.path.abspath(__file__)) + "/RVC/weights/"+voice_name+".pth")

    if os.path.exists(os.path.dirname(os.path.abspath(__file__)) + "/RVC/logs/"+voice_name):
      shutil.rmtree(os.path.dirname(os.path.abspath(__file__)) + "/RVC/logs/"+voice_name)

  for dataset in os.listdir(os.path.dirname(os.path.abspath(__file__)) + "/datasets/"+voice_name):
    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)) + "/datasets/"+voice_name, dataset)
    ds_rate, ds_data = wavfile.read(dataset_path)
    logging.info("Starting noise reduction process")
    if len(ds_data > 1):
      ds_data1 = ds_data[:,0]
      ds_data2 = ds_data[0:,1]
      reduced_noise1 = nr.reduce_noise(y=ds_data1, sr=ds_rate, chunk_size=32, use_torch=True, stationary=True)
      reduced_noise2 = nr.reduce_noise(y=ds_data2, sr=ds_rate, chunk_size=32, use_torch=True, stationary=True)
      reduced_noise = np.stack((reduced_noise1, reduced_noise2), axis=1)
    else:
      reduced_noise = nr.reduce_noise(y=ds_data, sr=ds_rate, chunk_size=32, use_torch=True, n_std_thresh_stationary=1,stationary=True)
      wavfile.write(dataset_path, rate, reduced_noise)
    logging.info("Starting silence remover process")
    remove_silence(2, dataset_path)

  os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/RVC/")
  os.system(f"python oneclickprocess.py --name {voice_name} --mode train --epochs " + str(epochs))
  os.chdir('../')

  if cleanup:
    if os.path.exists(os.path.dirname(os.path.abspath(__file__)) + "/RVC/logs/"+voice_name):
      shutil.rmtree(os.path.dirname(os.path.abspath(__file__)) + "/RVC/logs/"+voice_name)

    if os.path.exists(os.path.dirname(os.path.abspath(__file__)) + "/datasets/"+voice_name):
      shutil.rmtree(os.path.dirname(os.path.abspath(__file__)) + "/datasets/"+voice_name)

def talk(voice_name, text_prompt, job_id, language):
  global tts
  if not tts:
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)


  random_speaker_wav = random.choice(os.listdir(os.path.dirname(os.path.abspath(__file__)) + "/datasets/"+voice_name))
  speaker_wav = os.environ.get("TMP_DIR") + "/" + voice_name + "_speaker.wav"

  shutil.copy(random_speaker_wav, speaker_wav)

  filepath = os.environ.get("TMP_DIR") + "/" + voice_name + ".wav"

  tts.tts_to_file(text=text_prompt,
    file_path=filepath,
    speaker_wav=speaker_wav,
    language=language)


  
  if os.path.isfile(os.path.dirname(os.path.abspath(__file__)) + "/RVC/weights/"+voice_name+".pth"):
    #RVC Processing
    os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/RVC/")
    os.system(f"python oneclickprocess.py --name {voice_name} --mode generate --epochs 0")
    os.chdir('../')

  #os.chdir('./nuwave2')
  #os.system(f"python inference.py -c ../data/nuwave2.ckpt -i " + filepath + " --sr 24000")
  #os.chdir('../')

  final_path = os.environ.get("TMP_DIR") + "/" + job_id + ".wav"

  shutil.move(filepath, final_path)
  
  if os.path.exists(speaker_wav):
    os.remove(speaker_wav)

def request_audio(path):

  out = BytesIO()
  sound = AudioSegment.from_file(path)
  sound.export(out, format='mp3', bitrate="256k")
  out.seek(0)
  
  return out

def get_available_voices():
  foundvoices = {}
  for name in os.listdir(os.path.dirname(os.path.abspath(__file__)) + "/RVC/weights/"):
    if name != ".gitignore":
      voice_name = name.split(".")[0]
      if os.path.isdir(os.path.dirname(os.path.abspath(__file__)) + "/datasets/"+voice_name):
        foundvoices[voice_name] = voice_name
  return foundvoices