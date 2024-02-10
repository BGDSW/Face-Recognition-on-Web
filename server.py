from flask import Flask, request, render_template
from PIL import Image
import io
import base64
import argparse
import os
import cv2
import numpy as np
import torch

from backbones import get_model

app = Flask(__name__)


class FaceRecognition:
    def __init__(self):
        parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
        parser.add_argument('--network', type=str, default='r50', help='backbone network')
        parser.add_argument('--weight', type=str, default='')
        parser.add_argument('--img', type=str, default=None)
        args = parser.parse_args()
        args.network = 'vit_s'
        args.weight = './checkpoints/glint360k_model_TransFace_S.pt'

        self.net = get_model(args.network, fp16=False)
        self.net.load_state_dict(torch.load(args.weight))
        self.net.eval()

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        self.Cache = {}
        self.init_cache()



    @torch.no_grad()
    def init_cache(self):
        path = './imgs'
        directories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        for directory in directories:
            img_path = os.path.join(path, directory)
            file_list = []
            for root, dirs, files in os.walk(img_path):
                for file in files:
                    img_path = os.path.join(root, file)
                    feat = self.inference(img_path)
                    file_list.append(feat)
            self.Cache[directory] = file_list

    @torch.no_grad()
    def inference(self, img):
        if (img is None):
            img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
        elif (type(img) is str):
            img = cv2.imread(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        face_image = img
        for (x, y, w, h) in faces:
            face_image = img[y:y + h, x:x + w]
        img = cv2.resize(face_image, (112, 112))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).float()
        img.div_(255).sub_(0.5).div_(0.5)
        feat = self.net(img)
        return feat[0].view(-1).numpy()

    @torch.no_grad()
    def face_recognise(self, img, upload_name):
        img_np = np.array(img)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        feat = self.inference(img_np)
        name = ''
        if(upload_name == '#'):
            name = 'None'
            max_sim = -1
            for k in self.Cache.keys():
                for f in self.Cache[k]:
                    sim = np.dot(feat, f)
                    if (sim > 1 and sim > max_sim):
                        max_sim = sim
                        name = k
        else:
            name = upload_name + ' registered'
            if(upload_name in self.Cache.keys()):
                self.Cache[upload_name].append(feat)
            else:
                os.mkdir('./imgs/'+upload_name)
                self.Cache[upload_name] = [feat]
            img_path = './imgs/{}/{}.jpg'.format(upload_name, upload_name + str(len(self.Cache[upload_name])))
            cv2.imwrite(img_path, img_np)
        return name

model = FaceRecognition()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if 'name' not in request.form:
        return 'Enter name or #'
    name = request.form['name']

    if file:
        # Open and process the uploaded image
        img = Image.open(io.BytesIO(file.read()))
        # Process the image
        message = model.face_recognise(img, name)
        processed_img = img
        # Save processed image to a bytes buffer
        processed_img_io = io.BytesIO()
        processed_img.save(processed_img_io, format='JPEG')
        processed_img_io.seek(0)
        # Convert processed image to base64 string
        processed_img_base64 = base64.b64encode(processed_img_io.getvalue()).decode('utf-8')
        # Convert original image to base64 string
        original_img_base64 = base64.b64encode(file.read()).decode('utf-8')
        # Return both original and processed images as base64 strings, along with the name
        return render_template('index.html', name=message, original_image=original_img_base64,
                               processed_image=processed_img_base64)


if __name__ == '__main__':
    app.run(debug=True)
