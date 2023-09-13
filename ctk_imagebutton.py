import tkinter as tk
import customtkinter as ctk 
import os
from PIL import ImageTk
from authtoken import auth_token
from ctk_rangeslider import *
import PIL.Image
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, LMSDiscreteScheduler

import requests
from io import BytesIO
from huggingface_hub import notebook_login
from tkinter import filedialog
from tkinter import *
from os import path

class cTkImageButton(ctk.CTkButton):
    id=0
    denoise=0
    cfg=0
    image = None
    def setId(self,value):
        self.id=value
    def getId(self):
        return self.id
    
    def setDenoise(self,value):
        self.denoise=value
    def getDenoise(self):
        return self.denoise
    
    def setCFG(self,value):
        self.cfg=value
    def getCFG(self):
        return self.cfg
    
    def setImage(self,value):
        self.image=value
    def getImage(self):
        return self.image
    