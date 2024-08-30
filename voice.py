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

import torchaudio
import torch
from io import BytesIO
import psutil
from scipy.io import wavfile
import noisereduce as nr
from unsilence import Unsilence

import logging

from os.path import join, dirname
from dotenv import load_dotenv
dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)


logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=int(os.environ.get("LOG_LEVEL")),
        datefmt='%Y-%m-%d %H:%M:%S')

tts = None

def remove_silence(path, voice_name):
  nosilence_path = os.path.dirname(os.path.abspath(__file__)) + "/datasets/"+voice_name+"/sr_" + os.path.basename(path)
  logging.info("Detecting silence on: %s", path)
  u = Unsilence(path)
  u.detect_silence()
  logging.info("Removing silence on: %s", path)
  u.render_media(nosilence_path, audio_only=True)
  logging.info("Silenced removed saving, to file: %s", nosilence_path)
  if os.path.isfile(path):
    os.remove(path)
  return nosilence_path

def remove_noise(path, voice_name):
  ds_rate, ds_data = wavfile.read(path)
  logging.info("Starting noise reduction process on: %s", path)
  noise_reduced_path = os.path.dirname(os.path.abspath(__file__)) + "/datasets/"+voice_name+"/nr_" + os.path.basename(path)
  #if len(ds_data > 1):
  #  ds_data1 = ds_data[:,0]
  #  ds_data2 = ds_data[0:,1]
  #  reduced_noise1 = nr.reduce_noise(y=ds_data1, sr=ds_rate, chunk_size=32, use_torch=True, use_tqdm=True)
  #  reduced_noise2 = nr.reduce_noise(y=ds_data2, sr=ds_rate, chunk_size=32, use_torch=True, use_tqdm=True)
  #  reduced_noise = np.stack((reduced_noise1, reduced_noise2), axis=1)
  #  wavfile.write(noise_reduced_path, ds_rate, reduced_noise)
  #  logging.info("Noise reduction successfull, saving to file: %s", noise_reduced_path)
  #  if os.path.isfile(nosilence_path):
  #    os.remove(nosilence_path)
  #else:
  #  reduced_noise = nr.reduce_noise(y=ds_data, sr=ds_rate, chunk_size=32, use_torch=True, use_tqdm=True)
  #  wavfile.write(noise_reduced_path, ds_rate, reduced_noise)
  #  logging.info("Noise reduction successfull, saving to file: %s", noise_reduced_path)
  #  if os.path.isfile(nosilence_path):
  #    os.remove(nosilence_path)


  reduced_noise = nr.reduce_noise(y=ds_data, sr=ds_rate, chunk_size=32, use_torch=True, use_tqdm=True)
  wavfile.write(noise_reduced_path, ds_rate, reduced_noise)
  logging.info("Noise reduction successfull, saving to file: %s", noise_reduced_path)
  if os.path.isfile(path):
    os.remove(path)

def voice_clone(voice_name, epochs=100, dataset_paths=[]):

  if len(dataset_paths) > 0:
    if os.path.isfile(os.path.dirname(os.path.abspath(__file__)) + "/RVC/weights/"+voice_name+".pth"):
      os.remove(os.path.dirname(os.path.abspath(__file__)) + "/RVC/weights/"+voice_name+".pth")

    if os.path.exists(os.path.dirname(os.path.abspath(__file__)) + "/RVC/logs/"+voice_name):
      shutil.rmtree(os.path.dirname(os.path.abspath(__file__)) + "/RVC/logs/"+voice_name)
  
  for path in dataset_paths:
    logging.info("Converting to mono: %s", path)
    one_channel_path = os.path.dirname(os.path.abspath(__file__)) + "/datasets/"+voice_name+"/mono_" + os.path.basename(path)
    sound = AudioSegment.from_wav(path)
    sound = sound.set_channels(1)
    sound.export(one_channel_path, format="wav")
    logging.info("Mono file saved to: %s", one_channel_path)
    if os.path.isfile(path):
      os.remove(path)
    remove_noise(remove_silence(one_channel_path, voice_name), voice_name)

  os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/RVC/")
  os.system(f"python oneclickprocess.py --name {voice_name} --mode train --epochs " + str(epochs))
  os.chdir('../')

  checkpoints = []

  #for log_file in os.path.dirname(os.path.abspath(__file__)) + "/RVC/logs/"+voice_name:
  #  if log_file.endswith(".pth"):
  #    ck_path = os.path.join(os.path.dirname(os.path.abspath(__file__)) + "/RVC/logs/"+voice_name, log_file)
  #    if os.path.isfile(ck_path):
  #      checkpoints.append(ck_path)
  #      logging.info("Found checkpoint: %s", ck_path)

def talk(voice_name, text_prompt, job_id, language):
  global tts
  if tts is None:
    model = "tts_models/multilingual/multi-dataset/xtts_v2"
    logging.info("loading TTS model from: %s", model)
    tts = TTS(model, gpu=True)


  #path = os.path.dirname(os.path.abspath(__file__)) + "/RVC/logs/"+voice_name+"/1_16k_wavs/"
  path = os.path.dirname(os.path.abspath(__file__)) + "/datasets/"+voice_name+"/"
  random_speaker_wav = os.path.join(path, random.choice(os.listdir(path)))
  

  filepath = os.environ.get("TMP_DIR") + "/" + voice_name + ".wav"


  tts.tts_to_file(text=text_prompt,
    file_path=filepath,
    speaker_wav=random_speaker_wav,
    language=language)


  
  #RVC Processing
  os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/RVC/")
  os.system(f"python oneclickprocess.py --name {voice_name} --mode generate --epochs 0")
  os.chdir('../')

  os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/nuwave2')
  os.system(f"python inference.py -c ../data/nuwave2.ckpt -i " + filepath + " --sr 24000")
  os.chdir('../')

  final_path = os.environ.get("TMP_DIR") + "/" + job_id + ".wav"

  shutil.move(filepath, final_path)

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
      if os.path.isdir(os.path.dirname(os.path.abspath(__file__)) + "/RVC/logs/"+voice_name):
        foundvoices[voice_name] = voice_name
  return foundvoices