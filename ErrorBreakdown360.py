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
#model = ResNetMods.feedback_loop(model)
model.load_weights('./weights/50. 360NUbots - Res5 - train on 45Locations-4corners/my_weights_Res5_NU24')
model.compile(optimizer=Adam(lr=1e-4, epsilon=1e-10), loss=CustomImageGen.geo_loss, metrics=[CustomImageGen.xyz_error])


#####################################################################
# Test
x_test_folders = CustomImageGen.list_of_folders_from_file(".\\360cameraNUbotsField\\dataset4\\4Locations.txt")
x_test_files = CustomImageGen.list_of_files("NUbots360","test",folders=x_test_folders)

batch_size = 32
test_SPE = int(math.floor(len(x_test_files) / 32))

mycallback = CustomImageGen.MyMetrics(CustomImageGen.image_generator(x_test_files, batch_size, is_random=False, is_noise=False), test_SPE, batch_size)

model.evaluate(x=CustomImageGen.image_generator(x_test_files, batch_size, False, False), steps=test_SPE, callbacks=[mycallback])
file1 = open(".\\Results\\Errors.txt", "w")
errors = mycallback.get_all_errors()
for e in errors:
    file1.write("%s\n" % (e))
file1.close()

predictions = model.predict(x=CustomImageGen.image_generator(x_test_files, 32, False, False), steps=test_SPE, verbose=1)
np.savetxt(fname=".\\Results\\Predictions.txt", X=predictions, delimiter=",")
