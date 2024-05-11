import numpy as np
import unittest
from tensorflow.keras.preprocessing.image import load_img, img_to_array  # Correct import
from keras_vggface import VGGFace
from keras_vggface import utils
import keras

class VGGFaceTests(unittest.TestCase):
    def testVGG16(self):
        keras.backend.image_data_format()
        model = VGGFace(model='vgg16')
        img = load_img('image/ajb.jpg', target_size=(224, 224))
        x = img_to_array(img)  # Correct usage
        x = np.expand_dims(x, axis=0)
        x = utils.preprocess_input(x, version=1)
        preds = model.predict(x)
        self.assertIn('A.J._Buckley', utils.decode_predictions(preds)[0][0][0])
        self.assertAlmostEqual(utils.decode_predictions(preds)[0][0][1], 0.9790116, places=3)

    def testRESNET50(self):
        keras.backend.image_data_format()
        model = VGGFace(model='resnet50')
        img = load_img('image/ajb.jpg', target_size=(224, 224))
        x = img_to_array(img)  # Correct usage
        x = np.expand_dims(x, axis=0)
        x = utils.preprocess_input(x, version=2)
        preds = model.predict(x)
        self.assertIn('A._J._Buckley', utils.decode_predictions(preds)[0][0][0])
        self.assertAlmostEqual(utils.decode_predictions(preds)[0][0][1], 0.91819614, places=3)

    def testSENET50(self):
        keras.backend.image_data_format()
        model = VGGFace(model='senet50')
        img = load_img('image/ajb.jpg', target_size=(224, 224))
        x = img_to_array(img)  # Correct usage
        x = np.expand_dims(x, axis=0)
        x = utils.preprocess_input(x, version=2)
        preds = model.predict(x)
        self.assertIn('A._J._Buckley', utils.decode_predictions(preds)[0][0][0])
        self.assertAlmostEqual(utils.decode_predictions(preds)[0][0][1], 0.9993529, places=3)



def hicVGG16():
    keras.backend.image_data_format()
    model = VGGFace(model='vgg16')
    img = load_img('image/ajb.jpg', target_size=(224, 224))
    x = img_to_array(img)  # Correct usage
    x = np.expand_dims(x, axis=0)
    x = utils.preprocess_input(x, version=1)
    preds = model.predict(x)
    decoded_predictions = utils.decode_predictions(preds)[0][0]
    print('Identified Person:', decoded_predictions[0])
    print('Confidence:', decoded_predictions[1])
if __name__ == '__main__':
    # unittest.main()
    hicVGG16()