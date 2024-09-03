import os
import string
import random
import uuid
import shutil
from pydub import AudioSegment
from bark.generation import SAMPLE_RATE, preload_models, load_codec_model
from hubert.pre_kmeans_hubert import CustomHubert
from hubert.customtokenizer import CustomTokenizer

from bark.api import generate_audio
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
from unsilence import Unsilence

import logging

from os.path import join, dirname
from dotenv import load_dotenv
dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

tts = None
bark_preloaded = False

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

def clone_bark_voice(combined_sounds, voice_name):
  global bark_preloaded
  bark_to_clone_voice_path = os.environ.get("TMP_DIR") + "/" + str(uuid.uuid4().hex) + ".wav"
  combined_sounds.export(bark_to_clone_voice_path, format="wav")
  device = 'cuda' # or 'cpu'
  model = load_codec_model(use_gpu=True if device == 'cuda' else False)

  hubert_model = CustomHubert(checkpoint_path=os.path.dirname(os.path.abspath(__file__)) + '/data/models/hubert/hubert.pt').to(device)
  tokenizer = CustomTokenizer.load_from_checkpoint(os.path.dirname(os.path.abspath(__file__)) +'/data/models/hubert/model.pth').to(device)
  #audio_filepath = 'data/voices/' + voice_name + '.wav' # the audio you want to clone (under 13 seconds)
  wav, sr = torchaudio.load(bark_to_clone_voice_path)
  wav = convert_audio(wav, sr, model.sample_rate, model.channels)
  wav = wav.to(device)

  semantic_vectors = hubert_model.forward(wav, input_sample_hz=model.sample_rate)
  semantic_tokens = tokenizer.get_token(semantic_vectors)
  
  with torch.no_grad():
      encoded_frames = model.encode(wav.unsqueeze(0))
  codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze() 
  
  
  codes = codes.cpu().numpy()
  
  semantic_tokens = semantic_tokens.cpu().numpy()
  output_path = os.path.dirname(os.path.abspath(__file__)) +'/bark/assets/prompts/' + voice_name + '.npz'
  np.savez(output_path, fine_prompt=codes, coarse_prompt=codes[:2, :], semantic_prompt=semantic_tokens)
  bark_preloaded = False

  if os.path.isfile(bark_to_clone_voice_path):
    os.remove(bark_to_clone_voice_path)
  
def voice_clone(voice_name, job_id, epochs=100, dataset_paths=[], restart=False):
  try:
    logging.info("[%s] Starting voice clone process with %s epochs for voice_name: %s", job_id, str(epochs), voice_name)


    log_path = os.path.dirname(os.path.abspath(__file__)) + "/RVC/logs/"+voice_name

    if restart:
      if os.path.isfile(os.path.dirname(os.path.abspath(__file__)) + "/RVC/weights/"+voice_name+".pth"):
        os.remove(os.path.dirname(os.path.abspath(__file__)) + "/RVC/weights/"+voice_name+".pth")

      if os.path.exists(log_path):
        shutil.rmtree(log_path)
    
    if os.path.exists(log_path):
      for file in os.listdir(log_path):
        logfile_path = os.path.join(log_path, file)
        if os.path.isfile(logfile_path) and file.endswith(".log"):
          os.remove(logfile_path)

    combined_sounds = None
    

    for path in dataset_paths:
      logging.info("[%s] Converting to mono: %s", job_id, path)
      one_channel_path = os.path.dirname(os.path.abspath(__file__)) + "/datasets/"+voice_name+"/mono_" + os.path.basename(path)
      sound = AudioSegment.from_wav(path)
      sound = sound.set_channels(1)
      sound.export(one_channel_path, format="wav")
      logging.info("[%s] Mono file saved to: %s", job_id, one_channel_path)
      if os.path.isfile(path):
        os.remove(path)
      nosilence_file = remove_silence(one_channel_path, voice_name, job_id)
      sound = AudioSegment.from_wav(nosilence_file)
      if combined_sounds is None:
        combined_sounds = sound
      else:
        combined_sounds = combined_sounds = sound

    if combined_sounds is not None:
      clone_bark_voice(combined_sounds, voice_name)


    os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/RVC/")
    os.system(f"python oneclickprocess.py --name {voice_name} --mode train --epochs " + str(epochs))
    os.chdir('../')

    #checkpoints = []

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

def talk(voice_name, text_prompt, job_id, language, use_bark=False, use_rvc=True):

  filepath = os.environ.get("TMP_DIR") + "/" + voice_name + ".wav"

  if use_bark:
    global bark_preloaded
    if bark_preloaded is False:
      preload_models(path=os.path.dirname(os.path.abspath(__file__)) +'/bark/assets/prompts')
      bark_preloaded = True
    
    audio_array = generate_audio(text_prompt, history_prompt=voice_name, silent=False)

    wavfile.write(filepath, SAMPLE_RATE, audio_array)

  else:
    global tts
    if tts is None:
      model = "tts_models/multilingual/multi-dataset/xtts_v2"
      tts = TTS(model, gpu=True)


    #path = os.path.dirname(os.path.abspath(__file__)) + "/RVC/logs/"+voice_name+"/1_16k_wavs/"
    path = os.path.dirname(os.path.abspath(__file__)) + "/datasets/"+voice_name+"/"
    random_speaker_wav = os.path.join(path, random.choice(os.listdir(path)))
    
    tts.tts_to_file(text=text_prompt,
      file_path=filepath,
      speaker_wav=random_speaker_wav,
      language=language)

  if use_rvc:
    #RVC Processing
    os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/RVC/")
    os.system(f"python oneclickprocess.py --name {voice_name} --mode generate --epochs 0")
    os.chdir('../')

  #os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/nuwave2')
  #os.system(f"python inference.py -c ../data/nuwave2.ckpt -i " + filepath + " --sr 24000")
  #os.chdir('../')

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