import paho.mqtt.client as mqtt
from tensorflow.keras.models import load_model
import numpy as np
import json
from PIL import Image
from password_load import load_password

classes = ["daisy", "dandelion", "roses", "sunflowers", "tulips"] # used for test
cat_classes = ['Ragdolls', 'Singapura_cats', 'Persian_cats', 'Sphynx_cats', 'Pallas_cats']

FILENAME = 'modelpara/flowers.keras'  # used for test
EFFICIENTNET_FILENAME = 'modelpara/cat_classifier_efficientnet.h5'
XCEPTION_FILENAME = 'modelpara/cat_classifier_xception.h5'  # used for cat classification

class MQTTInferenceServer:
    """
    A simple MQTT inference server that listens for image classification requests.
    """
    def __init__(self, hostname, password_path="mqtt.pwd", model_path=FILENAME, classes=classes):
        self.classes = classes
        print("Loading model from ", model_path)
        try:
            self.model = load_model(model_path)
        except Exception as e:
            print("Error loading model:", e)
            try:
                self.model = load_model(model_path, compile=False)
            except Exception as e2:
                print("Error loading model without compilation:", e2)
                raise e2
        print("Done.")
        print("Connecting to broker")
        self.client = mqtt.Client()
        username, password = load_password(password_path)
        self.client.username_pw_set(username, password)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.connect(hostname)
        print("Done")


    def classify(self, image):
        result = self.model.predict(image)
        themax = np.argmax(result)
        print("Result: ", result)
        return (self.classes[themax], result[0][themax], themax)

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("Connected.")
            self.client.subscribe("Group19/IMAGE/classify")
        else:
            print("Failed to connect. Error code: ", rc)

    def classify_result(self, filename, data):
        print("Start classifying.")
        label, prob, index = self.classify(data)
        print("Done.")

        return {
            "filename": filename,
            "prediction": label,
            "score": float(prob),
            "index": str(index)
        }

    def on_message(self, client, userdata, msg):
        recv_dict = json.loads(msg.payload)
        img_data = np.array(recv_dict["data"])
        result = self.classify_result(recv_dict["filename"], img_data)
        print("Sending results.")
        self.client.publish("Group19/IMAGE/predict", json.dumps(result))
    
    def start_server(self):
        self.client.loop_start()
        while True:
            pass  # Keep the server running

if __name__ == "__main__":
    server = MQTTInferenceServer(hostname="127.0.0.1", classes=cat_classes, model_path=EFFICIENTNET_FILENAME)
    # server = MQTTInferenceServer(hostname="127.0.0.1", classes=classes, model_path=FILENAME)
    server.start_server()