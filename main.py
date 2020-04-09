
import numpy as np
import time

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import vgg19
from tensorflow.keras import backend as K

from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave

from mask import mask_image

BACKGROUND_PATH = 'background.jpg'
IMAGE_PATH = 'sample.jpeg'
STYLE_PATH = 'wave.jpg'

masked_image = mask_image(IMAGE_PATH, BACKGROUND_PATH)
width, height = masked_image.size
image_height = 400
image_width = int(width * image_height/height)

def preprocess_image(image):

    img = image.resize((image_height, image_height))
    img = img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    img = vgg19.preprocess_input(img)

    return img

def deprocess_image(x):
    
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    
    return x

class Evaluate(object):
    
    def __init__(self):
        
        self.loss_value = None
        self.grad_value = None
    
    def loss(self, x):
        
        assert self.loss_value is None
        x = x.reshape((1, image_height, image_width, 3))
        outs = fetch_loss_and_grads([x])
        loss_value = outs[0]
        grad_value = outs[1].flatten().astype('float64')
        self.loss_value = loss_value
        self.grad_value = grad_value
        
        return self.loss_value
    
    def grads(self, x):
        
        assert self.loss_value is not None
        grad_value = np.copy(self.grad_value)
        self.grad_value = None
        self.loss_value = None
        
        return grad_value

target_image = K.constant(preprocess_image(masked_image)
style_image = load_img(STYLE_PATH, target_size = (image_height, image_width))
style_ref_image = K.constant(preprocess_image(style_image))
combined_image = K.placeholder((1, image_height, image_width, 3))

input_tensor = K.concatenate([target_image, style_ref_image, combined_image], axis = 0)

model = vgg19.VGG19(input_tensor = input_tensor, weights = 'imagenet', include_top = False)

def content_loss(base, combination):
    return K.sum(K.square(combination - base))

def gram_matrix(x):
    
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    
    return gram


def style_loss(style, combination):
    
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = image_height * image_width
    
    return K.sum(K.square(S - C)) / (4.0 * (channels ** 2) * size ** 2)

def total_variation_loss(x):
    
    a = K.square(
                x[:, :image_height - 1, :image_width - 1, :] - 
                x[:, 1:, :image_width - 1, :])
    b = K.square(
                x[:, :image_height - 1, :image_width - 1, :] - 
                x[:, 1:, :image_height - 1, :])
    
    return K.sum(K.pow(a + b, 1.25))

outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
content_layer = 'block5_conv2'
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1',]
total_variation_weight = 1e-4
style_weight = 1.0
content_weight = 0.025


loss = K.variable(0.0)
layer_features = outputs_dict[content_layer]
target_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]

loss = loss + content_weight + content_loss(target_image_features, combination_features)

for layer_name in style_layers:
    
    layer_features = outputs_dict[layer_name]
    style_ref_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    
    sl = style_loss(style_ref_features, combination_features)
    
    loss = loss + (style_weight / len(style_layers)) * sl
    
loss = loss + total_variation_weight * total_variation_loss(combined_image)

grads = K.gradients(loss, combined_image)[0]
fetch_loss_and_grads = K.function([combined_image], [loss, grads])


evaluator = Evaluate()

iterations = 2

x = preprocess_image(target_path)
x = x.flatten()

for i in range(iterations):
    
    print("Iteration: ", i)
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x, fprime = evaluator.grads, maxfun = 20)
    print("Loss: ", min_val)
    img = x.copy().reshape((image_height, image_width, 3))
    img = deprocess_image(img)
    fname = "musk_waves" + "_at_iteration_%d.png" %i
    imsave(fname, img)
    print("Image Saved")