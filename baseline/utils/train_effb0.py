import cv2
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD
import os.path
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array

IMG_SIZE        = 224
BATCH_SIZE      = 32
EPOCHS_HEAD     = 10          # 先训练分类头
EPOCHS_FINE     = 20          # 再微调
TOP_N_TRAINABLE = 20          # 解冻的最顶层数
NUM_CLASSES     = 5
NOISE_LEVEL     = 0.1          # 高斯噪声水平
BLUR_LEVEL      = 3            # 模糊水平
BRIGHTNESS_RANGE = [1.0, 1.5]  # 亮度范围
MODEL_FILE = "cats_efficientnetb0-Noise-Brightness-V3-bright-05.keras"


def preview_augmentation(img_dir_path, datagen, num_images=9):
    """
    预览数据增强效果

    参数:
    img_path: 图片路径
    datagen: ImageDataGenerator实例
    num_images: 要显示的增强图片数量
    """
    if not os.path.exists(img_dir_path):
        raise FileNotFoundError(f"Image directory {img_dir_path} does not exist.")
    img_path = img_dir_path + os.listdir(img_dir_path)[0]
    img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # 创建图形
    plt.figure(figsize=(12, 12))

    # 显示原图
    plt.subplot(int(np.ceil((num_images + 1) / 3)), 3, 1)
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')

    # 显示增强后的图片
    i = 1
    for batch in datagen.flow(x, batch_size=1):
        plt.subplot(int(np.ceil((num_images + 1) / 3)), 3, i + 1)
        plt.imshow(batch[0].astype('uint8'))
        plt.title(f'Augmented #{i}')
        plt.axis('off')
        i += 1
        if i >= num_images:
            break

    plt.tight_layout

def add_gaussian_noise(image, noise_level=NOISE_LEVEL):
    noise = np.random.normal(0, noise_level, image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 255)

def add_blur(image, blur_level=BLUR_LEVEL):
    return cv2.GaussianBlur(image, (blur_level, blur_level), 0)

def add_noise_and_blur_preprocessing(image):
    image = add_gaussian_noise(image, NOISE_LEVEL)
    image = add_blur(image, BLUR_LEVEL)
    return image

def build_datasets():
    aug = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ])

    def _process(x, y, training=False):
        x = tf.cast(x, tf.float32)
        x = preprocess_input(x)               # ★ 统一预处理 (-1~1)
        if training:
            x = aug(x)
        return x, y

    raw_train = tf.keras.preprocessing.image_dataset_from_directory(
        "split_data/train",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode='int',
        shuffle=True
    )
    raw_val = tf.keras.preprocessing.image_dataset_from_directory(
        "split_data/val",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode='int',
        shuffle=False
    )

    class_names = raw_train.class_names
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = raw_train.map(lambda x, y: _process(x, y, training=True)).prefetch(AUTOTUNE)
    val_ds   = raw_val. map(lambda x, y: _process(x, y, training=False)).prefetch(AUTOTUNE)
    return train_ds, val_ds, class_names

def process_image_for_efficientnet(image_path):
    # 加载图像
    image = tf.keras.utils.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    
    # 转换为数组
    image_array = tf.keras.utils.img_to_array(image)
    
    # 添加批次维度
    image_array = tf.expand_dims(image_array, axis=0)
    
    # 关键：使用 EfficientNet 专用预处理
    processed_image = preprocess_input(image_array)
    
    return processed_image

def create_model(num_classes):
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base_model.trainable = False  # 初始冻结基础模型
    
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = preprocess_input(inputs)  # 关键：使用专用预处理
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    return model

def load_existing(model_file):
    model = load_model(model_file)
    # 解冻最后4个块进行微调
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):  # 找到基础模型
            base_model = layer
            break
    
    if base_model:
        # 解冻最后4个块
        for layer in base_model.layers[-20:]:
            if not isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = True
    return model

def train(model_file, train_path, validation_path, num_classes=5, steps=100, num_epochs=20):
    if os.path.exists(model_file):
        print("\n*** Loading existing model ***\n")
        model = load_existing(model_file)
        # 必须重新编译
        model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    else:
        print("\n*** Creating new model ***\n")
        model = create_model(num_classes)
        model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    checkpoint = ModelCheckpoint(
        model_file,
        save_best_only=True,
        monitor='val_accuracy',
        mode='max'
    )

    # 使用专用预处理
    train_datagen = ImageDataGenerator(
        preprocessing_function=add_noise_and_blur_preprocessing,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=BRIGHTNESS_RANGE,
        rotation_range=20
    )
    
    val_datagen = ImageDataGenerator(
        preprocessing_function=add_noise_and_blur_preprocessing,
        brightness_range=BRIGHTNESS_RANGE,
        rotation_range=20,
        shear_range=0.2
    )

    preview_augmentation(train_path+'/Pallas_cats/', train_datagen, num_images=4)

    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=32,  # 增大批量大小
        class_mode='sparse'
    )
    
    val_generator = val_datagen.flow_from_directory(
        validation_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=32,
        class_mode='sparse'
    )

    train_generator, val_generator,_ = build_datasets()

    # 第一阶段：训练新添加的层
    print("=== Phase 1: Training Head ===")
    history = model.fit(
        train_generator,
        steps_per_epoch=steps,
        epochs=num_epochs,
        callbacks=[checkpoint],
        validation_data=val_generator,
        validation_steps=20
    )

    # 第二阶段：微调
    print("\n=== Phase 2: Fine-Tuning ===")
    # 找到基础模型并解冻部分层
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            base_model = layer
            break
    
    if base_model:
        base_model.trainable = True
        # 解冻最后4个块 (EfficientNetB0有7个块，解冻block5b到block7a)
        for layer in base_model.layers:
            layer.trainable = False  # 先冻结所有
            
        # 解冻最后部分层
        for layer in base_model.layers[-20:]:
            if not isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = True

    # 使用更小的学习率
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        train_generator,
        steps_per_epoch=steps,
        epochs=num_epochs,
        callbacks=[checkpoint],
        validation_data=val_generator,
        validation_steps=20
    )

def main():
    train(
        MODEL_FILE,
        train_path="split_data/train",
        validation_path="split_data/val",
        steps=100,  # 根据数据集大小调整
        num_epochs=15
    )

if __name__ == '__main__':
    main()