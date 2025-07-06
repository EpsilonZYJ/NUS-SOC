from .server import MQTTInferenceServer

client = MQTTInferenceServer(hostname="127.0.0.1", model_name='efficientnet')