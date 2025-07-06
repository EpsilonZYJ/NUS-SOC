from .server import MQTTInferenceServer

client = MQTTInferenceServer(hostname="127.0.0.1", model_name='efficientnet')

model_dict = {
    "efficientnet": "model/cats_efficientnetb0-Noise-Brightness-V1.keras",
    "xception": "model/cat_classifier_xception.h5",
    "test": "model/flowers.keras",
    "insection": "model/cats_insection.keras"
}
