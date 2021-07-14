from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D,Input
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import CSVLogger

n_categories = 4
batch_size = 32
train_dir = 'train_casual'
validation_dir = 'validation_casual'
file_name = 'vgg16_sign_fine_casual_2'

base_model = VGG16(weights = 'imagenet',include_top=False,
                 input_tensor = Input(shape=(224,224,3)))

#add new layers instead of FC networks
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024,activation = 'relu')(x)
prediction = Dense(n_categories,activation = 'softmax')(x)
model = Model(inputs = base_model.input,outputs=prediction)

#fix weights before VGG16 14layers
for layer in base_model.layers[:15]:
    layer.trainable = False

model.compile(optimizer = SGD(lr=0.0001,momentum=0.9),
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

model.summary()

#save model
json_string=model.to_json()
open(file_name+'.json','w').write(json_string)

train_datagen = ImageDataGenerator(rescale=1./255,
                                    rotation_range=40,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    fill_mode="nearest")
 

validation_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size = (224,224),
    batch_size = batch_size,
    class_mode = 'categorical',
    shuffle = True
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size = (224,224),
    batch_size = batch_size,
    class_mode = 'categorical',
    shuffle = True
)
#・steps_per_epochとvalidation_stepsは、1エポックあたりのステップ数。すなわち、全サンプル数=バッチ数*ステップ数。(1426 = 32*step(45))
hist=model.fit(train_generator,
                         epochs = 10,
                         verbose = 1,
                         validation_steps = int(6),
                         validation_data = validation_generator,
                         steps_per_epoch = int(10),
                         callbacks=[CSVLogger(file_name + '.csv')])

#save weights
model.save(file_name + '.h5')