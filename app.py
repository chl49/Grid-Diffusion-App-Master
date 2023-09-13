import tkinter as tk
import customtkinter as ctk 
import os
from PIL import ImageTk
from authtoken import auth_token
from ctk_rangeslider import *
from ctk_imagebutton import *
import PIL.Image
import torch
from torch import autocast
from diffusers import StableDiffusionImg2ImgPipeline

import requests
from io import BytesIO
from huggingface_hub import notebook_login
from tkinter import filedialog
from tkinter import *
from os import path

url =  "https://avatarfiles.alphacoders.com/339/339938.jpg"
response = requests.get(url)
init_image = PIL.Image.open(BytesIO(response.content)).convert("RGB")
init_image.thumbnail((768, 768))
denoise = (0,1)
cfg = (0,20)
device = "cuda"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "nitrosocke/Ghibli-Diffusion", torch_dtype=torch.float16, use_auth_token=auth_token).to(device)

#url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.geometry("1500x900")
        self.title("Stable Bud") 
        ctk.set_appearance_mode("dark") 
        notebook_login()
        self.prompt_text = ctk.CTkLabel(self ,text="Prompt", text_color="white")
        self.prompt_text.place(x=30, y=20)
        self.prompt = ctk.CTkEntry(self, placeholder_text="Insert Prompt",height=40, width=480, text_color="black", fg_color="white") 
        self.prompt.place(x=30, y=50)
        
        self.generate_button = ctk.CTkButton(self, height=40, width=120, text_color="white", fg_color="blue", command=self.testImages) 
        self.generate_button.configure(text="Generate") 
        self.generate_button.place(x=530, y=50) 

        self.denoise_text = ctk.CTkLabel(self, text="Denoise Range:", text_color="white")
        self.denoise_text.place(x=30, y=110)
        self.denoise_min = ctk.CTkEntry(self, placeholder_text="0.0",height=30, width=45, text_color="black", fg_color="white") 
        self.denoise_min.insert(0, "0.0")
        self.denoise_min.place(x=130, y=110)
        self.denoise_max = ctk.CTkEntry(self, placeholder_text="1.0",height=30, width=45, text_color="black", fg_color="white") 
        self.denoise_max.insert(0, "1.0")
        self.denoise_max.place(x=180, y=110)
        self.denoise_slider = CTkRangeSlider(self, command=self.setDenoiseSlider, width = 200,height = 20, from_ = 0, to = 1, fg_color = "white")
        self.denoise_slider.place(x=30, y=160, relx=0, rely=0)
        
        self.cfg_text = ctk.CTkLabel(self ,text="CFG Range:", text_color="white")
        self.cfg_text.place(x=250, y=110)
        self.cfg_min = ctk.CTkEntry(self, placeholder_text="0.0",height=30, width=45, text_color="black", fg_color="white") 
        self.cfg_min.insert(0, "0.0")
        self.cfg_min.place(x=350, y=110)
        self.cfg_max = ctk.CTkEntry(self, placeholder_text="20.0",height=30, width=45, text_color="black", fg_color="white") 
        self.cfg_max.insert(0, "20.0")
        self.cfg_max.place(x=400, y=110)
        self.cfg_slider = CTkRangeSlider(self, command=self.setCFGSlider, width = 200,height = 20, from_ = 0, to = 20, fg_color = "white")
        self.cfg_slider.place(x=250, y=160, relx=0, rely=0)

        self.grid_text = ctk.CTkLabel(self ,text="Grid Size", text_color="white")
        self.grid_text.place(x=470, y=110)
        self.grid_entry = ctk.CTkEntry(self, placeholder_text="5",height=30, width=45, text_color="black", fg_color="white") 
        self.grid_entry.place(x=550, y=110)
        self.resample_button = ctk.CTkButton(self, height=40, width=120, text_color="white", fg_color="blue", command=self.resample) 
        self.resample_button.configure(text="Resample") 
        self.resample_button.place(x=530, y=150)
        
        
        self.browse_button = ctk.CTkButton(self, height=40, width=120, text_color="white", fg_color="blue", command=self.browse) 
        self.browse_button.configure(text="Upload Image") 
        self.browse_button.place(x=110, y=210) 
        
        
        self.branch_button = ctk.CTkButton(self, height=40, width=120, text_color="white", fg_color="blue", command=self.branch) 
        self.branch_button.configure(text="Save Image") 
        self.branch_button.place(x=420, y=210) 
        
        self.root_image = ctk.CTkLabel(self ,text="", fg_color = ("white", "gray75"), height=256, width=256)
        self.root_image.place(x=44, y=270)


        self.leaf_image = ctk.CTkLabel(self ,text="", fg_color = ("white", "gray75"), height=256, width=256)
        self.leaf_image.place(x=352, y=270)
        self.current_cfg =0
        self.current_denoise =0
        self.current_text = ctk.CTkLabel(self ,text="")
        self.current_text.place(x=352, y=540)
        self.current_image = None
        
        
        self.previous_cfg =0
        self.previous_denoise =0
        self.previous_text = ctk.CTkLabel(self ,text="")
        self.previous_text.place(x=352, y=560)
        
        
        self.frame = ctk.CTkFrame(self, width=700, height=700)
        self.frame.place(x=700, y=50) 
        
        self.cfg_index = []
        self.denoise_index = []
        self.image_buttons = []
        cfg_offset = 775
        denoise_offset = 110
        self.cfg_legend = ctk.CTkLabel(self ,text="", text_color="white")
        self.cfg_legend.place(x=1070, y=10) 
        self.denoise_legend = ctk.CTkLabel(self ,text="", wraplength=1, text_color="white")
        self.denoise_legend.place(x=650, y=340) 
        for i in range(5):
            row_buttons = []
            for j in range(5):
                button = cTkImageButton(self.frame , text = "", fg_color = ("transparent"), height=128, width=128)
                button.grid(row=i, column=j, padx=6, pady=6)
                button.configure( command= lambda btn= button: self.select(btn))
                button.setId((i*5)+(j))
                row_buttons.append(button)
            self.image_buttons.append(row_buttons)
            
            cfg_label = ctk.CTkLabel(self ,text="", text_color="white")
            cfg_label.place(x=cfg_offset, y=30) 
            self.cfg_index.append(cfg_label)
            cfg_offset = cfg_offset+155
            
            denoise_label = ctk.CTkLabel(self ,text="", text_color="white")
            denoise_label.place(x=665, y=denoise_offset) 
            self.denoise_index.append(denoise_label)
            denoise_offset = denoise_offset+148
        
    def setDenoiseSlider(self, values):
        global denoise
        denoise=values
        self.denoise_min.configure(placeholder_text="{}".format(values[0]))
        self.denoise_max.configure(placeholder_text="{}".format(values[1]))
        self.denoise_min.delete(0, END)
        self.denoise_min.insert(0, values[0]) 
        self.denoise_max.delete(0, END)
        self.denoise_max.insert(0, values[1]) 
    def setCFGSlider(self, values):
        global cfg
        cfg=values
        self.cfg_min.configure(placeholder_text="{}".format(values[0]))
        self.cfg_max.configure(placeholder_text="{}".format(values[1]))
        self.cfg_min.delete(0, END)
        self.cfg_min.insert(0, values[0]) 
        self.cfg_max.delete(0, END)
        self.cfg_max.insert(0, values[1]) 
    def browse(self):
        filename = filedialog.askopenfilename()
        folder_path = StringVar()
        folder_path.set(filename)
        global init_image
        init_image = PIL.Image.open(filename, 'r').convert("RGB")
        init_image.thumbnail((768, 768))
        img = ImageTk.PhotoImage(init_image.resize((256,256),PIL.Image.LANCZOS))
        self.root_image.configure(image=img) 
        print(filename)
    def select(self, bt):
        self.current_image=bt.getImage()
        self.leaf_image.configure(image=ImageTk.PhotoImage(self.current_image.resize((256,256),PIL.Image.LANCZOS))) 
        print(bt.getCFG())
        self.previous_cfg = self.current_cfg
        self.previous_denoise = self.current_denoise
        self.previous_text.configure(text = "previous settings: [ denoise: {}, CFG scale: {}]".format(self.previous_denoise, self.previous_cfg))
        self.current_cfg =bt.getCFG()
        self.current_denoise =bt.getDenoise()
        self.current_text.configure(text = "current settings: [ CFG scale: {}, denoise: {} ]".format(self.current_cfg, self.current_denoise))
        self.current_text.configure(text = "current settings: [ denoise: {}, CFG scale: {}]".format(self.current_denoise, self.current_cfg))
    def resample(self):
        
        self.cfg_min.configure(placeholder_text="{}".format(min(self.current_cfg, self.previous_cfg)))
        self.cfg_max.configure(placeholder_text="{}".format(max(self.current_cfg, self.previous_cfg)))
        
        self.denoise_min.configure(placeholder_text="{}".format(min(self.current_denoise, self.previous_denoise)))
        self.denoise_max.configure(placeholder_text="{}".format(max(self.current_denoise, self.previous_denoise)))
        
        self.cfg_min.delete(0, END)
        self.cfg_min.insert(0, min(self.current_cfg, self.previous_cfg))
        self.cfg_max.delete(0, END)
        self.cfg_max.insert(0, max(self.current_cfg, self.previous_cfg))
        
        self.denoise_min.delete(0, END)
        self.denoise_min.insert(0, min(self.current_denoise, self.previous_denoise))
        self.denoise_max.delete(0, END)
        self.denoise_max.insert(0, max(self.current_denoise, self.previous_denoise))
        
        
    def branch(self):
        file = filedialog.asksaveasfile(defaultextension=".png", filetypes=(("PNG file", "*.png"),("All Files", "*.*") ))
        if file:
            abs_path = os.path.abspath(file.name)
            self.current_image.save(abs_path)

    def testGenerate(self, denoise_val, cfg_val, col, row): 
        
        with autocast(device): 
            image = pipe(self.prompt.get(), image=init_image, strength=self.roundFloat(denoise_val), guidance_scale=self.roundFloat(cfg_val)).images[0]
            
        image1 = image.resize((128,128),PIL.Image.LANCZOS)
        
        img = ImageTk.PhotoImage(image1)
        self.image_buttons[col][row].configure(image=img) 
        self.image_buttons[col][row]._draw()
        self.image_buttons[col][row].setCFG(self.roundFloat(cfg_val)) 
        self.image_buttons[col][row].setDenoise(self.roundFloat(denoise_val)) 
        self.image_buttons[col][row].setImage(image)
        
    def testImages(self):
        denoise_inc=(self.dmaxRange(self.denoise_max.get())-self.minRange(self.denoise_min.get()))/4      
        cfg_inc=(self.cmaxRange(self.cfg_max.get())-self.minRange(self.cfg_min.get()))/4
        denoise_val = self.roundFloat(self.denoise_min.get())
        
        cfg_axis_val = self.roundFloat(self.cfg_min.get())
        denoise_sample = self.minRange(self.denoise_min.get())
        cfg_sample = self.minRange(self.cfg_min.get())
        
        self.cfg_legend.configure(text="CFG scale") 
        self.denoise_legend.configure(text="denoising")    
        self.current_cfg =cfg_sample
        self.current_denoise =denoise_sample
        self.current_text.configure(text = "current settings: [ denoise: {}, CFG scale: {}]".format(self.current_denoise, self.current_cfg))
        self.previous_text.configure(text = "")
        for i in range(5):
            
            cfg_val = self.roundFloat(self.cfg_min.get())
            self.cfg_index[i].configure(text=self.roundFloat(cfg_axis_val))
            self.denoise_index[i].configure(text=self.roundFloat(denoise_val))
            for j in range(5):
                self.testGenerate(denoise_val,cfg_val,i,j)
                cfg_val = cfg_val+cfg_inc
            denoise_val=denoise_val+denoise_inc
            cfg_axis_val=cfg_axis_val+cfg_inc
        image1 = self.image_buttons[0][0].getImage()
        self.leaf_image.configure(image=ImageTk.PhotoImage(image1.resize((256,256),PIL.Image.LANCZOS)))
                                
    def dmaxRange(self,val):
           _val= float(val)
           if _val > 1: _val = 1
           return float("{:.3f}".format(_val))
    def minRange(self,val):
           _val= float(val)
           if _val < 0: _val = 0
           print(_val)
           return float("{:.3f}".format(_val))
    def cmaxRange(self,val):
           _val= float(val)
           if _val > 20: _val = 20
           return float("{:.3f}".format(_val))
       
    def roundFloat(self,val):
            return float("{:.3f}".format(float(val)))
        
    def button_click(self):
        print("button click")

def show_value(value):
    print(value)
    

app = App()
app.mainloop()

