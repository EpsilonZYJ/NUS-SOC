import paho.mqtt.client as mqtt
import numpy as np
from PIL import Image
import json
from os import listdir
from os.path import join
from password_load import load_password

class MQTTMonitorClient:
    """
    A simple MQTT inference client that sends images for classification.
    """

    def __init__(self, hostname='0.0.0.0', password_path="mqtt.pwd"):
        self.client = mqtt.Client()
        username, password = load_password(password_path)
        self.client.username_pw_set(username, password)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.connect(hostname)

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("Connected.")
            self.client.subscribe("Group19/IMAGE/predict")
        else:
            print("Failed to connect. Error cde: ", rc)

    def on_message(self, client, userdata, msg):
        print("Received message from server.")
        resp_dict = json.loads(msg.payload)
        print("Filename: %s, Prediction: %s, Score: %3.4f)" % (resp_dict["filename"], resp_dict["prediction"],
                                                               float(resp_dict["score"])))

    def start_client(self):
        self.client.loop_start()
        print("Client started.")
        while True:
            pass


if __name__ == "__main__":
    client = MQTTMonitorClient()
    client.start_client()