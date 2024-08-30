import os
import string
import random
import uuid
import shutil
from pydub import AudioSegment
from encodec.utils import convert_audio
import numpy as np
import logging

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

tts = None

def remove_silence(path, voice_name, job_id):
  nosilence_path = os.path.dirname(os.path.abspath(__file__)) + "/datasets/"+voice_name+"/sr_" + os.path.basename(path)
  logging.info("[%s] Detecting silence on: %s", job_id, path)
  u = Unsilence(path)
  u.detect_silence()
  logging.info("[%s] Removing silence on: %s", job_id, path)
  u.render_media(nosilence_path, audio_only=True)
  logging.info("[%s] Silenced removed, saving to file: %s", job_id, nosilence_path)
  if os.path.isfile(path):
    os.remove(path)
  return nosilence_path

def remove_noise(path, voice_name, job_id):
  ds_rate, ds_data = wavfile.read(path)
  logging.info("[%s] Starting noise reduction process on: %s", job_id, path)
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
  logging.info("[%s] Noise reduction successful, saving to file: %s", job_id, noise_reduced_path)
  if os.path.isfile(path):
    os.remove(path)

def voice_clone(voice_name, job_id, epochs=100, dataset_paths=[]):
  try:
    logging.info("[%s] Starting voice clone process with %s epochs for voice_name: %s", job_id, str(epochs), voice_name)


    log_path = os.path.dirname(os.path.abspath(__file__)) + "/RVC/logs/"+voice_name

    if len(dataset_paths) > 0:
      if os.path.isfile(os.path.dirname(os.path.abspath(__file__)) + "/RVC/weights/"+voice_name+".pth"):
        os.remove(os.path.dirname(os.path.abspath(__file__)) + "/RVC/weights/"+voice_name+".pth")

      if os.path.exists(log_path):
        shutil.rmtree(log_path)
    
    if os.path.exists(log_path):
      for file in os.listdir(log_path):
        logfile_path = os.path.join(log_path, file)
        if os.path.isfile(logfile_path) and file.endswith(".log"):
          os.remove(logfile_path)
    
    for path in dataset_paths:
      logging.info("[%s] Converting to mono: %s", job_id, path)
      one_channel_path = os.path.dirname(os.path.abspath(__file__)) + "/datasets/"+voice_name+"/mono_" + os.path.basename(path)
      sound = AudioSegment.from_wav(path)
      sound = sound.set_channels(1)
      sound.export(one_channel_path, format="wav")
      logging.info("[%s] Mono file saved to: %s", job_id, one_channel_path)
      if os.path.isfile(path):
        os.remove(path)
      remove_noise(remove_silence(one_channel_path, voice_name, job_id), voice_name, job_id)

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
  except Exception as e:
    logging.error("[%s] FAIL!! voice_clone for %s failed.", job_id, voice_name)
    logging.error("[%s] Please check the server logs for further informations.", job_id, voice_name)
    raise e

def talk(voice_name, text_prompt, job_id, language):

  global tts
  if tts is None:
    model = "tts_models/multilingual/multi-dataset/xtts_v2"
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