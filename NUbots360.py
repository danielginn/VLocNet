from keras.applications import ResNet50
import ResNet50Modifications as ResNetMods
import CustomImageGen
from CustomImageGen import image_generator as img_gen
from keras.optimizers import Adam
import math
import random
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator


#####################################################################
# Load in Model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
print("ResNet50 model loaded...")
#for layer in model.layers[:143]: #175 is the final Activation layer: Activation_49, #143 is another one too.
#    layer.trainable = False
#    print(layer.name

model = ResNetMods.change_activation_function(model)
model = ResNetMods.additional_final_layers(model)
#model = ResNetMods.feedback_loop(model)
model.compile(optimizer=Adam(lr=1e-4, epsilon=1e-10), loss=CustomImageGen.geo_loss, metrics=[CustomImageGen.xyz_error])
model.summary()

#####################################################################
# Find Dataset
x_train_folders = CustomImageGen.list_of_folders_from_file(".\\360cameraNUbotsField\\dataset4smaller\\16Locations.txt")
x_test_folders = CustomImageGen.list_of_folders_from_file(".\\360cameraNUbotsField\\dataset4smaller\\33Locations.txt")
x_train_files = CustomImageGen.list_of_files("NUbots360","train",folders=x_train_folders)
x_test_files = CustomImageGen.list_of_files("NUbots360","test",folders=x_test_folders)

#####################################################################
# Train
batch_size = 32
train_SPE = int(math.floor(len(x_train_files)/batch_size))
test_SPE = int(math.floor(len(x_test_files)/batch_size))

file1 = open(".\\Results\\Results.txt", "w")
mycallback = CustomImageGen.MyMetrics(CustomImageGen.image_generator(x_test_files, batch_size, is_random=False, is_noise=False), test_SPE, batch_size)
results_train = model.fit(x=CustomImageGen.image_generator(x_train_files, batch_size, is_random=True, is_noise=True), steps_per_epoch=train_SPE, epochs=1, verbose=2, validation_data=CustomImageGen.image_generator(x_test_files, batch_size,is_random=False, is_noise=False), validation_steps=test_SPE, validation_freq=1, callbacks=[mycallback])
print(results_train.history)
print("mycallback:")
print(mycallback.get_median())
epoch_counter = 0
best_median_i = 0
file1.write("%s,%s,%s\n" % (epoch_counter, results_train.history["xyz_error"][0], mycallback.get_median()))
file1.close()
val_freq = 3

for i in range(26):
    print("train: Round " + str(i))
    results_train = model.fit(x=CustomImageGen.image_generator(x_train_files, batch_size, True, True), steps_per_epoch=train_SPE, epochs=val_freq, verbose=2, validation_data=CustomImageGen.image_generator(x_test_files, batch_size, False, False), validation_steps=test_SPE, validation_freq=1, callbacks=[mycallback])
    print("test:")
    median = mycallback.get_median()
    print(median)
    if i == 23:
        model.save_weights('my_weights_Res3_NU23')
    if i == 24:
        model.save_weights('my_weights_Res3_NU24')
    if i == 25:
        model.save_weights('my_weights_Res3_NU25')

    if (i == 16) or ((i >= 17) and (median < best_median)):
        best_median = median
        best_median_i = i
        model.save_weights('best_model')

    epoch_counter += val_freq
    file1 = open(".\\Results\\Results.txt", "a")
    file1.write("%s,%s,%s\n" % (epoch_counter, results_train.history["xyz_error"][0], mycallback.get_median()))
    file1.close()
    print("Best weights at: " + str(best_median_i))



