import os
import cv2
import numpy as np
import replicate
import threading
import argparse
import urllib.error
import urllib.request
from pythonosc import dispatcher
from pythonosc import osc_server
from pythonosc import udp_client
from pythonosc.osc_message_builder import OscMessageBuilder

count = 0
os.makedirs('./picture', exist_ok=True)
os.makedirs('./video', exist_ok=True)


def tex2img(prompt_input):
    global count
    #stable diffusionを使って画像をサーバー上で生成する。
    count += 1
    model = replicate.models.get("stability-ai/stable-diffusion")
    url = model.predict(prompt=prompt_input)[0]

    #生成した画像をurllibを使用しpngで保存。
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)
    picture_path = f'./picture/{count}.png'
    urllib.request.urlretrieve(url, picture_path)

    return picture_path

def img2vid(path):
    global count

    #cv2を使用して画像を10秒の動画に変換。fps等は適当な値で大丈夫。
    size = 256
    video_path = f'./video/{count}.mp4'

    #保存フォーマット指定。
    fourcc = cv2.VideoWriter_fourcc('m','p','4', 'v')
    video  = cv2.VideoWriter(video_path, fourcc, 20, (size,size))

    #リサイズしないと動かない。
    roop = int(20 * 10)
    img = cv2.imread(path)
    img = cv2.resize(img,(size,size))

    for i in range(roop):
        video.write(img)

    video.release()
    
    return video_path


