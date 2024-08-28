import os
import requests

f1 = './data/nuwave2.ckpt'
f2 = './data/f0D48k.pth'
f3 = './data/f0G48k.pth'
#f4 = './data/hubert_base.pt'

def get_model():
    if not os.path.isfile(f1) or not os.path.isfile(f2) or not os.path.isfile(f3):
        url1 = 'https://github.com/ORI-Muchim/BARK-RVC/releases/download/v1.0/f0D48k.pth'
        url2 = 'https://github.com/ORI-Muchim/BARK-RVC/releases/download/v1.0/f0G48k.pth'
        #rl3 = 'https://github.com/ORI-Muchim/BARK-RVC/releases/download/v1.0/hubert_base.pt'
        url4 = 'https://github.com/ORI-Muchim/BARK-RVC/releases/download/v1.0/nuwave2.ckpt'

        print("Downloading Pretrained Discriminator Model...")
        response1 = requests.get(url1, allow_redirects=True)

        print("Downloading Pretrained Generator Model...")
        response2 = requests.get(url2, allow_redirects=True)
        
        #print("Downloading Hubert Model...")
        #response3 = requests.get(url3, allow_redirects=True)
        
        print("Downloading Nu-Wave2 Model...")
        response4 = requests.get(url4, allow_redirects=True)

        directory = './data'
        #directory2 = './RVC'

        pretrained_discriminator_model = os.path.join(directory, 'f0D48k.pth')
        pretrained_generator_model = os.path.join(directory, 'f0G48k.pth')
        hubert_model = os.path.join(directory, 'hubert_base.pt')
        nuwave2_model = os.path.join(directory, 'nuwave2.ckpt')
        
        with open(pretrained_discriminator_model, 'wb') as file:
            file.write(response1.content)
        print("Saving Pretrained Discriminator Model...")

        with open(pretrained_generator_model, 'wb') as file:
            file.write(response2.content)
        print("Saving Pretrained Generator Model...")
        
        #with open(hubert_model, 'wb') as file:
        #    file.write(response3.content)
        #print("Saving Hubert Model...")
        
        with open(nuwave2_model, 'wb') as file:
            file.write(response4.content)
        print("Saving NU-Wave2 Model...\n")