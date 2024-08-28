import os
import shutil
from os.path import join, dirname
from dotenv import load_dotenv
from hubert.hubert_manager import HuBERTManager
from get_model import get_model

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

hubert_manager = HuBERTManager()
hubert_manager.make_sure_hubert_installed()
hubert_manager.make_sure_tokenizer_installed()

get_model()

tmpdir_path =  os.environ.get("TMP_DIR") + "/"
if os.path.exists(tmpdir_path): 
  for filename in os.listdir(tmpdir_path):
    file_path = os.path.join(tmpdir_path, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))