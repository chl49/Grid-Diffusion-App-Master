# Grid Diffusion Application
Image Generation Application made for directed studies project: https://www.youtube.com/watch?v=b3u-5_nwI3M&list=WL

## Features
- img2img generation using prompt specification
- range selection for denoising (strength) and CFG scale (guidance) using sliders
- resampling of denoising strength and CFG scale parameters by selecting 2 images 

## Setup
1. download repository
- clone git repository with https://github.com/chl49/Grid-Diffusion-App-Master.git
2. install dependencies
- pip install Pillow
- pip install numpy
- pip install packaging
- pip install torch
- pip install torchaudio
- pip install torchvision
- pip install transformers
- pip install cuda-python
- pip install tkinterdnd2
- pip install customtkinter
- pip install --upgrade diffusers[torch]
- pip install auth-token
- pip install --upgrade huggingface_hub
- pip install 'huggingface_hub[cli,torch]'
- pip install 'huggingface_hub[tensorflow]'
3. run application
- python app.py

## Demo
https://www.youtube.com/watch?v=J8w6RHrMUFE
