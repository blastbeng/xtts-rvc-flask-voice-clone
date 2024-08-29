import os
import string
import random
import uuid
import shutil
from pydub import AudioSegment
from encodec.utils import convert_audio
import numpy as np
from silenceremove import process as remove_silence

from IPython.display import Audio


from TTS.api import TTS 

from transformers import AutoProcessor, BarkModel


import torchaudio
import torch
from io import BytesIO
import psutil
from scipy.io import wavfile
import noisereduce as nr


from os.path import join, dirname
from dotenv import load_dotenv
dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

tts = False

def voice_clone(voice_name, audio_filepath, epochs=100, cleanup=False, from_scratch=False):

  if from_scratch:
    if os.path.isfile(os.path.dirname(os.path.abspath(__file__)) + "/RVC/weights/"+voice_name+".pth"):
      os.remove(os.path.dirname(os.path.abspath(__file__)) + "/RVC/weights/"+voice_name+".pth")

    if os.path.exists(os.path.dirname(os.path.abspath(__file__)) + "/RVC/logs/"+voice_name):
      shutil.rmtree(os.path.dirname(os.path.abspath(__file__)) + "/RVC/logs/"+voice_name)

  if audio_filepath is not None:
    rate, data = wavfile.read(audio_filepath)
    reduced_noise = nr.reduce_noise(y=data, sr=rate)
    wavfile.write(audio_filepath, rate, reduced_noise)
    remove_silence(audio_filepath)

  for dataset in os.listdir(os.path.dirname(os.path.abspath(__file__)) + "/datasets/"):
    dataset_path = os.path.join(directory, dataset)
    ds_rate, ds_data = wavfile.read(dataset_path)
    reduced_noise = nr.reduce_noise(y=ds_data, sr=ds_rate)
    wavfile.write(dataset_path, rate, reduced_noise)
    remove_silence(dataset_path)

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


  filepath = os.environ.get("TMP_DIR") + "/" + voice_name + ".wav"

  tts.tts_to_file(text=text_prompt,
    file_path=filepath,
    speaker_wav=os.path.dirname(os.path.abspath(__file__)) + "/voices/"+voice_name+".wav",
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
  
  #if os.path.exists(filepath):
  #  os.remove(filepath)

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
      foundvoices[name.split(".")[0]] = name.split(".")[0]
  return foundvoices