import csv
import os
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import cv2
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import PIL.Image

## Global variable
model = None  #model
db = None
image_path = ""
id = ""
fn = ""
value = ""

def angle_error(y_true, y_pred):
    """
    Calculate the mean diference between the true angles
    and the predicted angles. Each angle is represented
    as a binary vector.
    """
    diff = angle_difference(K.argmax(y_true), K.argmax(y_pred))
    return K.mean(K.cast(K.abs(diff), K.floatx()))

# Download model file from cloud storage bucket
def download_model_file():

    from google.cloud import storage

    # Model Bucket details
    BUCKET_NAME        = "model-predict"
    PROJECT_ID         = "careful-muse-313003"
    GCS_MODEL_FILE     = "rotnet_barcode_view_resnet50_v2.hdf5"

    # Initialise a client
    client   = storage.Client(PROJECT_ID)
    
    # Create a bucket object for our bucket
    bucket   = client.get_bucket(BUCKET_NAME)
    
    # Create a blob object from the filepath
    blob     = bucket.blob(GCS_MODEL_FILE)
    
    folder = '/tmp/'
    if not os.path.exists(folder):
        os.makedirs(folder)
    # Download the file to a destination
    blob.download_to_filename(folder + "model.hdf5")

def download_image(event, context):

    from google.cloud import storage

    file = event
    # Model Bucket details
    BUCKET_NAME        = "careful-muse-313003.appspot.com"
    PROJECT_ID         = "careful-muse-313003"
    GCS_IMAGE_FILE     = file['name']

    # Initialise a client
    client   = storage.Client(PROJECT_ID)
    
    # Create a bucket object for our bucket
    bucket   = client.get_bucket(BUCKET_NAME)
    
    # Create a blob object from the filepath
    blob     = bucket.blob(GCS_IMAGE_FILE)
    
    folder = '/tmp/'
    if not os.path.exists(folder):
        os.makedirs(folder)
    # Download the file to a destination
    global image_path
    image_path = folder + GCS_IMAGE_FILE
    blob.download_to_filename(image_path)

    #adding function split name
    global fn
    global id
    fn= file['name']
    id = fn.split('-') #use id[-2]


def farmacode(event, context):
    
    #download image to predict
    download_image(event, context)

    #deploy model locally
    global model
    if not model:
        download_model_file()
        model = tf.keras.models.load_model('/tmp/model.hdf5', custom_objects={'angle_error': angle_error})
    
    
    #initialize firestore
    global db
    if not db:
        # Use the application default credentials
        cred = credentials.ApplicationDefault()
        firebase_admin.initialize_app(cred, {
            'projectId': 'careful-muse-313003',
        })
    
    db = firestore.client()

    #load image
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    #predict
    global value
    x = np.vstack([x])
    value = np.argmax(model.predict(x), axis=1)

    

    #update firestore
    doc_ref = db.collection(u'farmacode-classification').document(id[-2])
    doc_ref.set({
        fn.strip('.jpg').strip('.jpeg').strip('.png'): value
    }, merge=True)
    print(fn + '\n')
    print(value + '\n')