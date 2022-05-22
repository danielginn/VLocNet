from keras.applications import ResNet50
import ResNet50Modifications as ResNetMods
import CustomImageGen
from CustomImageGen import image_generator as img_gen
from keras.optimizers import Adam
import math
import random
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

random.seed(123)

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
# Uncomment for NUbotsField
#x_train_files = CustomImageGen.list_of_files("NUbots","Cropped\\21Locations-plus","nothing")
#x_test_files = CustomImageGen.list_of_files("NUbots","Cropped\\4Locations-plus","nothing")

# Uncomment for 360CameraNUbotsField
#train_folders_file = ".\\360cameraNUbotsField\\dataset4\\16Locations.txt"
#test_folders_file = ".\\360cameraNUbotsField\\dataset4\\33Locations.txt"
#x_train_files = CustomImageGen.list_of_files("360cameraNUbotsField", None, CustomImageGen.list_of_folders_from_file(train_folders_file))
#x_test_files = CustomImageGen.list_of_files("360cameraNUbotsField", None, CustomImageGen.list_of_folders_from_file(test_folders_file))

# Uncomment for BlenderNUbotsField
train_folders_file = ".\\BlenderRoboCupTrain31\\all_locations.txt"
test_folders_file = ".\\FieldPositions56TestGrid\\all_locations.txt"
x_train_files = CustomImageGen.list_of_files("BlenderRoboCup", None, CustomImageGen.list_of_folders_from_file(train_folders_file))
x_test_files = CustomImageGen.list_of_files("BlenderRoboCup", None, CustomImageGen.list_of_folders_from_file(test_folders_file))

#####################################################################
# Training
batch_size = 4
train_SPE = int(math.floor(len(x_train_files)/batch_size))
test_SPE = int(math.floor(len(x_test_files)/batch_size))

file1 = open(".\\Results\\Results.txt", "w")
mycallback = CustomImageGen.MyMetrics(CustomImageGen.image_generator(x_test_files, batch_size, is_random=False, is_noise=False), test_SPE, batch_size)
results_train = model.fit(x=CustomImageGen.image_generator(x_train_files, batch_size, is_random=True, is_noise=False), steps_per_epoch=train_SPE, epochs=1, verbose=2, validation_data=CustomImageGen.image_generator(x_test_files, batch_size, False, False), validation_steps=test_SPE, validation_freq=1, callbacks=[mycallback])
print(results_train.history)
print("mycallback:")
print(mycallback.get_median())
epoch_counter = 0
best_median_i = 0
best_median10_i = 0
best_median15_i = 0
best_median20_i = 0
best_median25_i = 0
file1.write("%s,%s,%s\n" % (epoch_counter, results_train.history["xyz_error"][0], mycallback.get_median()))
file1.close()
val_freq = 3

for i in range(0,30):
    print("train: Round " + str(i))
    results_train = model.fit(x=CustomImageGen.image_generator(x_train_files, batch_size, True, False), steps_per_epoch=train_SPE, epochs=val_freq, verbose=2, validation_data=CustomImageGen.image_generator(x_test_files, batch_size, False, False), validation_steps=test_SPE, validation_freq=1, callbacks=[mycallback])
    print("test:")
    median = mycallback.get_median()
    print(median)

    if (i == 0) or ((i >= 1) and (median < best_median)):
        best_median = median
        best_median_i = i
        model.save_weights('my_weights_best_model_at_'+str(i))

    if (i == 10) or ((i >= 11) and (median < best_median10)):
        best_median10 = median
        best_median10_i = i
        model.save_weights('my_weights_best_model_from10')

    if (i == 15) or ((i >= 16) and (median < best_median15)):
        best_median15 = median
        best_median15_i = i
        model.save_weights('my_weights_best_model_from15')

    if (i == 20) or ((i >= 21) and (median < best_median20)):
        best_median20 = median
        best_median20_i = i
        model.save_weights('my_weights_best_model_from20')

    if (i == 25) or ((i >= 26) and (median < best_median25)):
        best_median25 = median
        best_median25_i = i
        model.save_weights('my_weights_best_model_from25')

    epoch_counter += val_freq
    file1 = open(".\\Results\\Results.txt", "a")
    file1.write("%s,%s,%s\n" % (epoch_counter, results_train.history["xyz_error"][0], mycallback.get_median()))
    file1.close()
    print("Best weights from 11 at: " + str(best_median10_i))
    print("Best weights from 16 at: " + str(best_median15_i))
    print("Best weights from 21 at: " + str(best_median20_i))
    print("Best weights from 26 at: " + str(best_median25_i))



