# example of training an unconditional gan on the fashion mnist dataset
import numpy as np
from numpy.random import randn
from numpy.random import randint
from keras.optimizers import Adam
from keras import backend
from keras.models import Sequential
from keras.layers import MaxPooling2D, Input, Dense, Reshape, UpSampling2D, Concatenate, Flatten, Conv2D, Activation, \
  Conv2DTranspose, LeakyReLU, Dropout, BatchNormalization, concatenate
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.initializers import RandomNormal
import os
from PIL import Image
from image_preprocessing import preprocess
from keras.models import Model
from numpy import savez_compressed, load
import matplotlib.pyplot as plt


def load_images(src_path='src_images/', tar_path='tar_images/', type='train', size=(512, 512)):
  type = './' + type + '/'
  src_list, tar_list = list(), list()
  # enumerate filenames in directory, assume all are images
  for filename in os.listdir(type + src_path):
    # load and resize the image
    # convert to numpy array
    img = Image.open(type + src_path + filename)
    img = img.convert('1')
    pixels = np.array(img)[:,:,np.newaxis]
    # split into satellite and map
    src_list.append(pixels)
  for filename in os.listdir(type + tar_path):
    # load and resize the image
    # pixels = np.array(Image.open(type + tar_path + filename))
    img = Image.open(type + tar_path + filename)
    img = img.convert('1')
    pixels = np.array(img)[:,:,np.newaxis]
    # split into satellite and map
    tar_list.append(pixels)
  return (np.asarray(src_list), np.asarray(tar_list))


def define_discriminator(image_shape=(512, 512, 1)):
  # weight initialization
  init = RandomNormal(stddev=0.02)
  # source image input
  in_src_image = Input(shape=image_shape)
  # target image input
  in_target_image = Input(shape=image_shape)
  # concatenate images channel-wise
  merged = Concatenate()([in_src_image, in_target_image])
  # TODO
  new_conv = Conv2D(filters=64,
                    kernel_size=(3, 3),
                    padding='same', kernel_initializer=init)(merged)
  vgg_model = VGG16(weights='imagenet', include_top=False)
  layers = [l for l in vgg_model.layers]
  for i in range(3):  # TODO
    layers.pop(0)
  for i in range(6):  # TODO
    layers.pop()
  # d = BatchNormalization()(new_conv) # TODO
  d = new_conv
  for i in range(len(layers)):
    layers[i].trainable = False
    d = layers[i](d)
  d = BatchNormalization()(d)
  d = LeakyReLU(alpha=0.2)(d)
  d = Conv2D(1, (4, 4), padding='same', kernel_initializer=init)(d)
  # patch output
  # patch_out = LeakyReLU(alpha=0.4)(d)
  patch_out = Activation(activation='tanh')(d)  # TODO
  # define model
  model = Model([in_src_image, in_target_image], patch_out)
  # compile model
  opt = Adam(lr=0.0003, beta_1=0.5)  # TODO
  model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
  return model


# define an encoder block
def define_encoder_block(layer_in, n_filters, batchnorm=True):
  # weight initialization
  init = RandomNormal(stddev=0.02)
  # add downsampling layer
  g = Conv2D(n_filters, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(layer_in)  # TODO
  # conditionally add batch normalization
  if batchnorm:
    g = BatchNormalization()(g, training=True)  # TODO
  # leaky relu activation
  g = LeakyReLU(alpha=0.2)(g)  # TODO
  return g

# define a decoder block
def decoder_block(layer_in, skip_in, n_filters, dropout=True):
  # weight initialization
  init = RandomNormal(stddev=0.02)
  # add upsampling layer
  g = Conv2DTranspose(n_filters, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(layer_in)
  # add batch normalization
  g = BatchNormalization()(g, training=True)
  # conditionally add dropout
  if dropout:
    g = Dropout(0.5)(g, training=True)
  # merge with skip connection
  g = Concatenate()([g, skip_in])
  # relu activation
  # g = LeakyReLU(alpha=0.2)(g)  # TODO
  g = Activation('relu')(g)
  return g


# define the standalone generator model
def define_generator(image_shape=(512, 512, 1)):
  # weight initialization
  init = RandomNormal(stddev=0.02)
  # image input
  in_image = Input(shape=image_shape)
  # encoder model
  e1 = define_encoder_block(in_image, 64, batchnorm=False)  # TODO: batch_norm=False
  e2 = define_encoder_block(e1, 128, batchnorm=True)
  e3 = define_encoder_block(e2, 256, batchnorm=True)
  e4 = define_encoder_block(e3, 512, batchnorm=True)
  e5 = define_encoder_block(e4, 512, batchnorm=True)
  e6 = define_encoder_block(e5, 512, batchnorm=True)
  e7 = define_encoder_block(e6, 512, batchnorm=True)
  # bottleneck, no batch norm and relu
  b = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(e7)
  # b = Activation('relu')(b)  # TODO
  b = LeakyReLU(alpha=0.1)(b)
  # b = Activation(activation='tanh')(b)
  # decoder model
  d1 = decoder_block(b, e7, 512)
  d2 = decoder_block(d1, e6, 512)
  d3 = decoder_block(d2, e5, 512)
  d4 = decoder_block(d3, e4, 512, dropout=False)
  d5 = decoder_block(d4, e3, 256, dropout=False)
  d6 = decoder_block(d5, e2, 128, dropout=False)
  d7 = decoder_block(d6, e1, 64, dropout=False)
  # output
  g = Conv2DTranspose(image_shape[2], (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d7)
  out_image = Activation('tanh')(g)  # TODO
  # define model
  model = Model(in_image, out_image)
  return model


def define_gan(g_model, d_model, image_shape=(512, 512, 1)):
  # make weights in the discriminator not trainable
  d_model.trainable = False
  # define the source image
  in_src = Input(shape=image_shape)
  # connect the source image to the generator input
  gen_out = g_model(in_src)
  # connect the source input and generator output to the discriminator input
  dis_out = d_model([in_src, gen_out])
  # src image as input, generated image and classification output
  model = Model(in_src, [dis_out, gen_out])
  # compile model
  opt = Adam(lr=0.0003, beta_1=0.5)  # TODO
  model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1, 100])
  return model


def load_real_samples(filename):
  # load compressed arrays
  data = load(filename)
  # unpack arrays
  X1, X2 = data['arr_0'], data['arr_1']
  # scale from [0,255] to [-1,1]
  X1 = (X1 - 127.5) / 127.5
  X2 = (X2 - 127.5) / 127.5
  return [X1, X2]


# select a batch of random samples, returns images and target
def generate_real_samples(dataset, n_samples, patch_shape):
  # unpack dataset
  trainA, trainB = dataset
  # choose random instances
  ix = randint(0, trainA.shape[0], n_samples)
  # retrieve selected images
  X1, X2 = trainA[ix], trainB[ix]
  # generate 'real' class labels (1)
  y = np.ones((n_samples, patch_shape, patch_shape, 1))
  return [X1, X2], y


def generate_fake_samples(g_model, samples, patch_shape):
  # generate fake instance
  X = g_model.predict(samples)
  # create 'fake' class labels (0)
  y = np.zeros((len(X), patch_shape, patch_shape, 1))
  return X, y


def summarize_performance(step, g_model, dataset, n_samples=4):
  # select a sample of input images
  [X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1)
  # generate a batch of fake samples
  X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
  # scale all pixels from [-1,1] to [0,1]
  X_realA = (X_realA + 1) / 2.0
  X_realB = (X_realB + 1) / 2.0
  X_fakeB = (X_fakeB + 1) / 2.0
  X_realA = X_realA[:, :, :, 0]
  X_realB = X_realB[:, :, :, 0]
  X_fakeB = X_fakeB[:, :, :, 0]
  # plot real source images
  for i in range(n_samples):
    plt.subplot(3, n_samples, 1 + i)
    plt.axis('off')
    plt.imshow(X_realA[i], cmap='gray')
  # plot generated target image
  for i in range(n_samples):
    plt.subplot(3, n_samples, 1 + n_samples + i)
    plt.axis('off')
    plt.imshow(X_fakeB[i], cmap='gray')
  # plot real target image
  for i in range(n_samples):
    plt.subplot(3, n_samples, 1 + n_samples * 2 + i)
    plt.axis('off')
    plt.imshow(X_realB[i], cmap='gray')
  # save plot to file
  filename1 = 'plot_%06d.png' % (step + 1)
  plt.savefig(filename1)
  plt.close()
  # save the generator model
  filename2 = 'model_%06d.h5' % (step + 1)
  g_model.save(filename2)
  print('>Saved: %s and %s' % (filename1, filename2))


def generate(src_test, tar_test):
  import random
  id = random.randrange(0, len(src_test))
  sketch = src_test[id]
  target = tar_test[id]
  transformed, _ = generate_fake_samples(g_model, sketch[np.newaxis, :, :, :], 1)
  sketch = (sketch + 1) / 2.0
  target = (target + 1) / 2.0
  transformed = (transformed + 1) / 2.0
  f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
  ax1.imshow(sketch, cmap='gray')
  ax2.imshow(transformed[0])
  ax3.imshow(target)
  plt.show()


# train pix2pix models
def train(d_model, g_model, gan_model, dataset, test_data, start_iter=0, n_epochs=100, n_batch=1):
  # determine the output square shape of the discriminator
  n_patch = d_model.output_shape[1]
  # unpack dataset
  trainA, trainB = dataset
  # calculate the number of batches per training epoch
  bat_per_epo = int(len(trainA) / n_batch)
  # calculate the number of training iterations
  n_steps = bat_per_epo * n_epochs
  # manually enumerate epochs
  for i in range(start_iter, n_steps):
    # select a batch of real samples
    [X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)
    # generate a batch of fake samples
    X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
    # update discriminator for real samples
    d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
    # update discriminator for generated samples
    d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
    # update the generator
    g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
    # summarize performance
    print('>%d, l_real[%.3f] l_fake[%.3f] g[%.3f]' % (i + 1, d_loss1, d_loss2, g_loss))
    # summarize model performance
    # if (i + 1) % (bat_per_epo // 4) == 0:
    if (i + 1) % (30) == 0:
      summarize_performance(i, g_model, test_data)


N = 1
image_size = 512
n_channel = 3

# preprocess(image_size, image_size, type='train')

type = 'train'
compress_arr_fn = type + '_' + str(image_size) + '.npz'
src_path = 'src_images/'
tar_path = 'tar_images/'
# src_images, tar_images = load_images(src_path=src_path, tar_path=tar_path, type=type, size=(image_size, image_size))
# print('Loaded Source Images: ', src_images.shape, '|| Loaded Target Images: ', tar_images.shape)
# savez_compressed(compress_arr_fn, src_images, tar_images)
# print("Saved to:", compress_arr_fn)
#
# type = 'test'
# compress_arr_fn = type + '_' + str(image_size) + '.npz'
# src_path = 'src_images/'
# tar_path = 'tar_images/'
# src_images, tar_images = load_images(src_path=src_path, tar_path=tar_path, type=type, size=(image_size, image_size))
# print('Loaded Source Images: ', src_images.shape, '|| Loaded Target Images: ', tar_images.shape)
# savez_compressed(compress_arr_fn, src_images, tar_images)
# print("Saved to:", compress_arr_fn)

# data = load(compress_arr_fn)
# src_images, tar_images = data['arr_0'], data['arr_1']
# print(np.max(src_images))
data = load_real_samples('train_512.npz')
test_data = load_real_samples('test_512.npz')
src_images, tar_images = data
print('Loaded Train Source Images: ', data[0].shape, '|| Loaded Train Target Images: ', data[1].shape)
print('Loaded Test Source Images: ', test_data[0].shape, '|| Loaded Test Target Images: ', test_data[1].shape)
## src_images, tar_images = np.load('src'+compress_arr_fn), np.load('tar'+compress_arr_fn)


import random
id = random.randrange(0, src_images.shape[0])
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
print(src_images[id,:, : ,0].shape)
ax1.imshow(src_images[id,:, : ,0], cmap='gray')
ax2.imshow(tar_images[id,:, : ,0], cmap='gray')
plt.show()

image_shape = src_images.shape[1:]

# image_shape = (512, 512, 3)
# define discriminator and generator
d_model = define_discriminator(image_shape)
g_model = define_generator(image_shape)
start_iter = 0
# try:
#   g_model.load_weights('model_000' + str(start_iter) + '.h5')
# except:
#   pass
# d_model.summary()
# g_model.summary()
# define GAN
gan_model = define_gan(g_model, d_model, image_shape)
# train model
train(d_model, g_model, gan_model, data, test_data, start_iter=start_iter, n_batch=4)

# src_test, tar_test = load_real_samples('test_512.npz')
# generate(src_test, tar_test)

# test_data = load_real_samples('test_512.npz')
# summarize_performance(0,g_model, test_data)
