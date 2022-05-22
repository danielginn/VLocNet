from keras.applications import ResNet50
import ResNet50Modifications as ResNetMods
import CustomImageGen
from CustomImageGen import image_generator as img_gen
from keras.optimizers import Adam
import math
import numpy as np

#####################################################################
# Load in Model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = ResNetMods.change_activation_function(model)
model = ResNetMods.additional_final_layers(model)
model.load_weights('./weights/402. BlenderRoboCupTrain31 - Res3/my_weights_best_model_from25')
model.compile(optimizer=Adam(lr=1e-4, epsilon=1e-10), loss=CustomImageGen.geo_loss, metrics=[CustomImageGen.xyz_error])
model.summary()

#####################################################################
# Test
# Uncomment for NUbotsField
#x_test_files = CustomImageGen.list_of_files("NUbots","cropped/16Locations-plus","nothing")

# Uncomment for 360CameraNUbotsField
#test_folders_file = "./360cameraNUbotsField/dataset4/33Locations.txt"
#x_test_files = CustomImageGen.list_of_files("360cameraNUbotsField", None, CustomImageGen.list_of_folders_from_file(test_folders_file))

# Uncomment for 360CameraNUbotsField
test_folders_file = "./BlenderRoboCupPaths/path5.txt"
x_test_files = CustomImageGen.list_of_files("BlenderRoboCup", None, CustomImageGen.list_of_folders_from_file(test_folders_file))

batch_size = 4
test_SPE = int(math.floor(len(x_test_files)/batch_size))

mycallback = CustomImageGen.MyMetrics(CustomImageGen.image_generator(x_test_files, batch_size, is_random=False, is_noise=False), test_SPE, batch_size)

model.evaluate(x=CustomImageGen.image_generator(x_test_files, batch_size, False, False), steps=test_SPE, callbacks=[mycallback])
file1 = open("./Results/Errors.txt", "w")
errors = mycallback.get_all_errors()
for e in errors:
    file1.write("%s\n" % (e))
file1.close()

predictions = model.predict(x=CustomImageGen.image_generator(x_test_files, batch_size, False, False), steps=test_SPE, verbose=1)
np.savetxt(fname="./Results/Predictions.txt", X=predictions, delimiter=",")
