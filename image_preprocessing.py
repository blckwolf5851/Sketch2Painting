import numpy as np
from PIL import Image, ImageEnhance
import scipy.ndimage
import cv2
import os
from smartcrop import SmartCrop
import numpy as np
import argparse
import glob
import cv2
from PIL import ImageFilter
from albumentations.augmentations import transforms

def grayscale(rgb):
  return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


size = (512,512,3)


def laplacianFilter(alpha):
  laplacian = np.array([
    [alpha / 4, (1 - alpha) / 4, alpha / 4],
    [(1 - alpha) / 4, -1, (1 - alpha) / 4],
    [alpha / 4, (1 - alpha) / 4, alpha / 4]
  ])
  laplacian = (4 / (alpha + 1)) * laplacian
  return laplacian

def dodge(front, back):
  result=front*255/(255-back+1e-10)
  result[result>255]=255
  result[back==255]=255
  return result.astype('uint8')


def sketch_from_image(in_path, out_path):
  count = 0
  for filename in os.listdir(in_path):
    image = Image.open(in_path+filename)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.5)
    edges = Image.fromarray(edge_detect(np.array(image)))
    gray_img = grayscale(np.array(image))
    # print(edge_detect(np.array(image)).shape)
    inverted_img = 255 - gray_img
    blur_img = scipy.ndimage.filters.gaussian_filter(inverted_img, sigma=7)
    final_img = dodge(blur_img, gray_img)
    final_img = Image.fromarray(final_img)
    enhancer = ImageEnhance.Contrast(final_img)
    final_img = enhancer.enhance(2)
    # final_img.show()
    edges.putalpha(136)
    final_img.paste(edges, (0, 0), edges)
    final_img = cv2.fastNlMeansDenoising(np.array(final_img), None, 10, 7, 30)
    # if count % 6 == 0:
    #   cv2.imshow('sketch', final_img)
    #   cv2.waitKey(0)
    cv2.imwrite(out_path + filename, final_img)
    count += 1
    # final_img.save(out_path + filename)


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
  # initialize the dimensions of the image to be resized and
  # grab the image size
  dim = None
  (h, w) = image.shape[:2]
  # if both the width and height are None, then return the
  # original image
  if width is None and height is None:
    return image
  # make the short side to satisfy the square requirement
  if h < w:
    # calculate the ratio of the height and construct the
    # dimensions
    r = height / float(h)
    dim = (int(w * r), height)

  # otherwise, the height is None
  else:
    # calculate the ratio of the width and construct the
    # dimensions
    r = width / float(w)
    dim = (width, int(h * r))

  # resize the image
  resized = cv2.resize(image, dim, interpolation=inter)

  # return the resized image
  return resized


def prepare_src_image(in_path, out_path, w, h):
  for filename in os.listdir(in_path):
    try:
      image = cv2.imread(in_path + filename)
      image = image_resize(image,w,h)
      filename = '.'.join(filename.split('.')[:-1])
      cv2.imwrite(out_path + filename+'.jpg', image)
    except:
      continue

def auto_crop(in_path, out_path, w, h):
  sc = SmartCrop()
  for filename in os.listdir(in_path):
    image = Image.open(in_path + filename)
    crop = sc.crop(image, w, h)['top_crop']
    x, y = crop['x'], crop['y']
    image.crop((x, y, x+w, y+h)).save(out_path + filename)

def edge_detect(image, sigma=0.33):
  # image = scipy.ndimage.filters.gaussian_filter(image, sigma=1.5)
  image = cv2.fastNlMeansDenoising(np.array(image), None, 10, 7, 21)
  image = cv2.medianBlur(image,5)
  image = cv2.bilateralFilter(image, 9, 50, 50)
  # image = Image.fromarray(image, mode='RGB').filter(ImageFilter.EDGE_ENHANCE)
  # image = np.array(image)
  v = np.median(image)
  # apply automatic Canny edge detection using the computed median
  lower = int(max(0, (1.0 - sigma) * v))
  upper = int(min(255, (1.0 + sigma) * v))
  edged = 255 - cv2.Canny(image, lower, upper)
  img = Image.fromarray(edged)
  img = img.convert("RGBA")
  data = img.getdata()
  newData = []
  for item in data:
    if item[0] > 200 and item[1] > 200 and item[2] > 200:
      newData.append((255, 255, 255, 0))
    else:
      newData.append(item)
  # return the edged image
  img.putdata(newData)
  return np.array(img)

def augmentation(in_path='./train/raw_images/', out_path='./train/aug_images/'):
  for filename in os.listdir(in_path):
    hf = transforms.HorizontalFlip(always_apply=True)
    vf = transforms.VerticalFlip(always_apply=True)
    tp = transforms.Transpose(always_apply=True)
    rt = transforms.Rotate(limit=80, always_apply=True)
    image = np.array(Image.open(in_path + filename))
    hf_image = hf(image=image)['image']
    vf_image = vf(image=image)['image']
    tp_image = tp(image=image)['image']
    rt_image = rt(image=image)['image']
    count = 1
    for img in [image, hf_image, vf_image, tp_image, rt_image]:
      if len(img.shape) == 2:
        img = Image.fromarray(img)
        img.convert(mode='RGB')
      else:
        img = Image.fromarray(img, mode='RGB')
      img.save(out_path + filename.replace('.jpg', '_'+str(count)+'.jpg'))
      count += 1

def preprocess_sketch(in_path, out_path):
  for filename in os.listdir(in_path):
    sketch = cv2.imread(in_path + filename)
    dst = cv2.fastNlMeansDenoising(sketch, None, 10, 7, 21)
    cv2.imshow('original', sketch)
    cv2.imshow('cleaned', dst)
    cv2.waitKey(0)

def preprocess(w, h, type='train'):
  root_dir = './' + type + '/'
  if type == 'train':
    print('Augment Data')
    augmentation(root_dir + 'raw_images/', root_dir + 'aug_images/')
    print("Resize Image")
    prepare_src_image(root_dir+'aug_images/', root_dir+'resized_images/', w, h)
    print("Crop")
    auto_crop(root_dir+'resized_images/', root_dir+'tar_images/', w, h)
    print("Generate Sketch")
    sketch_from_image(root_dir+'tar_images/', root_dir+'src_images/')
  else:
    print("Resize Image")
    prepare_src_image(root_dir+'raw_images/', root_dir+'resized_images/', w, h)
    print("Crop")
    auto_crop(root_dir+'resized_images/', root_dir+'tar_images/', w, h)
    print("Generate Sketch")
    sketch_from_image(root_dir+'tar_images/', root_dir+'src_images/')



if __name__ == '__main__':
  preprocess(512, 512, 'usage')
  # preprocess_sketch('./train/src_images/', './train/src_images/')

  '''
  https://www.kaggle.com/c/painter-by-numbers/data
  https://www.reddit.com/r/MachineLearning/comments/6ecfa2/d_art_datasets/
  '''
