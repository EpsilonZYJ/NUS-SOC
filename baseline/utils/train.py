from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B3
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
import tensorflow as tf
import os.path

MODEL_FILE = "EfficientNetV2B3-V2.keras"

def create_model(num_hidden, num_classes, input_shape=(249, 249, 3)):
    # base_model = InceptionV3(include_top=False, weights='imagenet')
    base_model = EfficientNetV2B3(include_top=False, weights='imagenet', input_shape=input_shape)
    # x = base_model.output
    # x = GlobalAveragePooling2D()(x)
    # x = Dense(num_hidden, activation='relu')(x)
    # predictions = Dense(num_classes, activation='softmax')(x)
    # for layer in base_model.layers:
    #     layer.trainable = False
    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)

    return model

def load_existing(model_file):
    model = load_model(model_file)
    numlayers = len(model.layers)

    for layer in model.layers[:numlayers-5]:
        layer.trainable = False
    for layer in model.layers[numlayers-5:]:
        layer.trainable = True

    return model

def train(model_file, train_path, validation_path, num_hidden=200, num_classes=5, steps=32, num_epochs=20):
    if os.path.exists(model_file):
        print("\n*** Existing model found at %s. Loading.***\n\n" % model_file)
        model = load_existing(model_file)
    else:
        print("\n*** Creating new model ***\n\n")
        model = create_model(num_hidden, num_classes)

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    checkpoint = ModelCheckpoint(model_file)

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(249, 249),
        batch_size=5,
        class_mode='categorical'
    )
    validation_generator = test_datagen.flow_from_directory(
        validation_path,
        target_size=(249, 249),
        batch_size=5,
        class_mode='categorical'
    )

    model.fit(
        train_generator,
        steps_per_epoch=steps,
        epochs=num_epochs,
        callbacks=[checkpoint],
        validation_data=validation_generator,
        validation_steps=50
    )

    for layer in model.layers[:-15]:
        layer.trainable = False
    for layer in model.layers[-15:]:
        layer.trainable = True

    model.compile(
        optimizer=SGD(learning_rate=0.00001, momentum=0.9),
        loss='categorical_crossentropy'
    )

    model.fit(
        train_generator,
        steps_per_epoch=steps,
        epochs=num_epochs,
        callbacks=[checkpoint],
        validation_data=validation_generator,
        validation_steps=50
    )

def main():
    train(
        MODEL_FILE,
        train_path="split_data/train",
        validation_path="split_data/val",
        steps=120,
        num_epochs=20
    )

if __name__ == '__main__':
    main()