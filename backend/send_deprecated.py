import paho.mqtt.client as mqtt
import numpy as np
from PIL import Image
import json
from os import listdir
from os.path import join
from password_load import load_password

PATH = "./samples" # used for test
CAT_PATH = "./cat_samples"

EFFICIENTNET_INPUT_SIZE = (224, 224)
XCEPTION_INPUT_SIZE = (299, 299)

class MQTTInferenceClient:
    """
    A simple MQTT inference client that sends images for classification.
    """
    def __init__(self, hostname='0.0.0.0', password_path="mqtt.pwd", img_path=PATH, size=(249, 249)):
        self.client = mqtt.Client()
        username, password = load_password(password_path)
        self.client.username_pw_set(username, password)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.connect(hostname)
        self.send_cnt = 0
        self.img_path = img_path

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("Connected.")
            self.client.subscribe("Group19/IMAGE/predict")
        else:
            print("Failed to connect. Error cde: ", rc)

    def on_message(self, client, userdata, msg):
        print("Received message from server.")
        resp_dict = json.loads(msg.payload)
        print("Filename: %s, Prediction: %s, Score: %3.4f)" % (resp_dict["filename"], resp_dict["prediction"], float(resp_dict["score"])))
        self.send_cnt -= 1


    def load_image(self, filename, size=(249, 249)):
        img = Image.open(filename)
        img = img.resize(size)
        imgarray = np.asarray(img)/255.0
        final = np.expand_dims(imgarray, axis=0)
        return final

    def send_image(self, client, filename, size=(249, 249)):
        img = self.load_image(filename=filename, size=size)
        img_list = img.tolist()
        send_dict = {"filename": filename, "data": img_list}
        self.client.publish("Group19/IMAGE/classify", json.dumps(send_dict))

    def start_sending(self, size=(249, 249)):
        print("Sending data.")
        self.send_cnt = 0
        for file in listdir(self.img_path):
            filename = join(self.img_path, file)
            self.send_image(self.client, filename, size=size)
            self.send_cnt += 1
        print("Done. Waiting for results")
        while self.send_cnt > 0:
            pass
    
    def start_client(self, size=(249, 249)):
        self.client.loop_start()
        print("Client started.")
        while True:
            self.start_sending(size=size)

if __name__ == "__main__":
    # client = MQTTInferenceClient(img_path=PATH)
    # client.start_client(size=(249, 249))

    client = MQTTInferenceClient(img_path=CAT_PATH)
    client.start_client(size=EFFICIENTNET_INPUT_SIZE)  # Use EFFICIENTNET_INPUT_SIZE for EfficientNet model
    # To stop the client gracefully, you can use a keyboard interrupt (Ctrl+C)
    # or implement a shutdown mechanism in the loop.