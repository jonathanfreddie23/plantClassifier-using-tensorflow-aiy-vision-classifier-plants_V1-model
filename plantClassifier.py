import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd

# Load the pre-trained model
m = hub.KerasLayer('https://tfhub.dev/google/aiy/vision/classifier/plants_V1/1')

# Load and preprocess an image for classification
image_path = 'C:/Users/jonat/Desktop/python/Single_flower02.jpg'

image = tf.io.read_file(image_path)
image = tf.image.decode_image(image, channels=3)
image = tf.image.convert_image_dtype(image, tf.float32)
image = tf.image.resize(image, (224, 224))

# Make sure your image is in the correct shape (e.g., [1, 224, 224, 3])
input_image = tf.expand_dims(image, axis=0)

# Perform classification
output = m(input_image)

# Read the class names and IDs from a CSV file
csv_file = 'C:/Users/jonat/Desktop/python/labelmap.csv'  # Replace with the path to your CSV file
df = pd.read_csv(csv_file)

# Create the class mapping dictionary
class_mapping = dict(zip(df['id'], df['name']))

# Get the predicted class index
predicted_class_index = tf.argmax(output, axis=1).numpy()[0]

# Get the corresponding plant name from the class mapping
predicted_plant_name = class_mapping.get(predicted_class_index, "Unknown Plant")

# Print the predicted plant name
print("Predicted Plant:", predicted_plant_name)
