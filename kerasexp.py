import numpy as np
import matplotlib.pyplot as plt
plt.ion()
from PIL import Image
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras import Model, layers
from keras.models import load_model, model_from_json

input_path = 'data/'

train_datagen = ImageDataGenerator(
    shear_range=10,
    zoom_range=0.2,
    horizontal_flip=True,
    preprocessing_function=preprocess_input)
 
train_generator = train_datagen.flow_from_directory(
    input_path+'train',
    batch_size=32,
    class_mode='binary',
    target_size=(224,224))
 
validation_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input)
 
validation_generator = validation_datagen.flow_from_directory(
    input_path+'validation',
    shuffle=False,
    class_mode='binary',
    target_size=(224,224))

conv_base = ResNet50(include_top=False,
                     weights='imagenet')
 
for layer in conv_base.layers:
    layer.trainable = False
 
x = conv_base.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
predictions = layers.Dense(2, activation='softmax')(x)
model = Model(conv_base.input, predictions)
 
optimizer = keras.optimizers.Adam()
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

history = model.fit_generator(
    generator=train_generator,
    epochs=3,
    validation_data=validation_generator,
    steps_per_epoch=100 // 12,
    validation_steps=10)

# architecture and weights to HDF5
model.save('models/keras/model.h5')
 
# architecture to JSON, weights to HDF5
model.save_weights('models/keras/weights.h5')
with open('models/keras/architecture.json', 'w') as f:
    f.write(model.to_json())

# architecture and weights from HDF5
model = load_model('models/keras/model.h5')
 
# architecture from JSON, weights from HDF5
with open('models/keras/architecture.json') as f:
    model = model_from_json(f.read())
model.load_weights('models/keras/weights.h5')

validation_img_paths = ["validation/cats/cat-10.jpg",
                        "validation/dogs/dog-10.jpg",
                        "validation/cats/cat-03.jpg",
                        "validation/dogs/dog-05.jpg"]
img_list = [Image.open(input_path + img_path) for img_path in validation_img_paths]

validation_batch = np.stack([preprocess_input(np.array(img.resize((224,224))))
                             for img in img_list])

pred_probs = model.predict(validation_batch)
pred_probs

fig, axs = plt.subplots(1, len(img_list), figsize=(20, 5))
for i, img in enumerate(img_list):
    ax = axs[i]
    ax.axis('off')
    ax.set_title("{:.0f}% Cat, {:.0f}% Dog".format(100*pred_probs[i,0],
                                                            100*pred_probs[i,1]))
    ax.imshow(img)

plt.pause(30)