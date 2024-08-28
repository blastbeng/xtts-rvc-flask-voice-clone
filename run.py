import os
import sys
import scipy.io.wavfile

from bark import SAMPLE_RATE, generate_audio, preload_models
from get_model import get_model


preload_models(path="bark/assets/prompts")

if os.path.exists("barkrvc_out.wav"):
  os.remove("barkrvc_out.wav")

#Bark Processing
preload_models()

model_name = "giulio88ita"

text_prompt = """ Devo cinquanta mila euro a Fabio, appena torno a Torino gli faccio un bonifico """

audio_array = generate_audio(text_prompt, history_prompt=model_name)

scipy.io.wavfile.write("./out/barkrvc_out.wav", SAMPLE_RATE, audio_array)


#RVC Processing
get_model()

os.chdir('./RVC')
os.system(f"python oneclickprocess.py --name {model_name}")


#NU-Wave2 Processing
os.chdir('../nuwave2')
os.system(f"python inference.py -c ./data/nuwave2.ckpt -i ../out/rvc_out.wav --sr 24000")
