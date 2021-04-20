from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Construct an ImageDataGenerator object:

training_data_generator = ImageDataGenerator(   
  rescale=1/255,
   #Randomly increase or decrease the size of the image by up to 20%
        zoom_range=0.2, 

        #Randomly rotate the image between -15,15 degrees
        rotation_range=15, 

        #Shift the image along its width by up to +/- 5%
        width_shift_range=0.05, 

        #Shift the image along its height by up to +/- 5%
        height_shift_range=0.05 )