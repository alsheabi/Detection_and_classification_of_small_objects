import cv2
import numpy as np
class Augmenter_grayscale(object):
    def __call__(self, sample,gs=0.5):



      """ imgae grayscale """

      if np.random.rand() < gs: # random aug image
        image, annots = sample['img'], sample['annot']
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        stacked_img = np.stack((gray,)*3, axis=-1) 
        sample = {'img': stacked_img, 'annot': annots}    
        
      return sample

class Augmenter_hue(object):
    def __call__(self, sample,  hgain=0.5, sgain=0.0, vgain=0.0,hsv=0.5):



      """ HSV color-space augmentation
      Hue represents the color in range [0.179]  with gain +/- 50 degree
      
      """

      if np.random.rand() < hsv:
        image, annots = sample['img'], sample['annot']
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        dtype = image.dtype
        # print(dtype)
        # print(type(dtype))
        hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.uint8))

        x = np.arange(0, 256, dtype=dtype)
        lut_hue = ((x * r[0]) % 180)
        lut_sat = np.clip(x * r[1], 0, 255)
        lut_val = np.clip(x * r[2], 0, 255)
        hue = cv2.LUT(hue, lut_hue)
        sat = cv2.LUT(sat, lut_sat)
        val = cv2.LUT(val, lut_val)        
        
        image = cv2.merge((hue, sat, val)).astype(dtype)
        image= cv2.cvtColor(image, cv2.COLOR_HSV2BGR) 
        sample = {'img': image, 'annot': annots}    
      return sample
class Augmenter_saturation(object):
    def __call__(self, sample,  hgain=0.0, sgain=0.5, vgain=0.0,hsv=0.5):



      """ HSV color-space augmentation
      Saturation represents the greyness in range[ 0,255]  with gain +/- 50%
      """

      if np.random.rand() < hsv:
        image, annots = sample['img'], sample['annot']
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        dtype = image.dtype
        # print(dtype)
        # print(type(dtype))
        hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.uint8))

        x = np.arange(0, 256, dtype=dtype)
        lut_hue = ((x * r[0]) % 180)
        lut_sat = np.clip(x * r[1], 0, 255)
        lut_val = np.clip(x * r[2], 0, 255)
        hue = cv2.LUT(hue, lut_hue)
        sat = cv2.LUT(sat, lut_sat)
        val = cv2.LUT(val, lut_val)        
        
        image = cv2.merge((hue, sat, val)).astype(dtype)
        image= cv2.cvtColor(image, cv2.COLOR_HSV2BGR) 
        sample = {'img': image, 'annot': annots}    
      return sample
class Augmenter_value(object):
    def __call__(self, sample,  hgain=0.0, sgain=0.0, vgain=0.5,hsv=0.5):



      """ HSV color-space augmentation
      Value represents the brightness in range [0,255] with gain +/- 50%
      """
      if np.random.rand() < hsv:
        image, annots = sample['img'], sample['annot']
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        dtype = image.dtype
        # print(dtype)
        # print(type(dtype))
        hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.uint8))

        x = np.arange(0, 256, dtype=dtype)
        lut_hue = ((x * r[0]) % 180)
        lut_sat = np.clip(x * r[1], 0, 255)
        lut_val = np.clip(x * r[2], 0, 255)
        hue = cv2.LUT(hue, lut_hue)
        sat = cv2.LUT(sat, lut_sat)
        val = cv2.LUT(val, lut_val)        
        
        image = cv2.merge((hue, sat, val)).astype(dtype)
        image= cv2.cvtColor(image, cv2.COLOR_HSV2BGR) 
        sample = {'img': image, 'annot': annots}    
      return sample
class Augmenter_sv(object):
    def __call__(self, sample,  hgain=0.0, sgain=0.5, vgain=0.5,hsv=0.5):



      """ HSV color-space augmentation
       image merge stauration (image grayness with gain +/- 50% ) and value ( image brightness with gain +/- 50%)
      """

      if np.random.rand() < hsv:
        image, annots = sample['img'], sample['annot']
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        dtype = image.dtype
        # print(dtype)
        # print(type(dtype))
        hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.uint8))

        x = np.arange(0, 256, dtype=dtype)
        lut_hue = ((x * r[0]) % 180)
        lut_sat = np.clip(x * r[1], 0, 255)
        lut_val = np.clip(x * r[2], 0, 255)
        hue = cv2.LUT(hue, lut_hue)
        sat = cv2.LUT(sat, lut_sat)
        val = cv2.LUT(val, lut_val)        
        
        image = cv2.merge((hue, sat, val)).astype(dtype)
        image= cv2.cvtColor(image, cv2.COLOR_HSV2BGR) 
        sample = {'img': image, 'annot': annots}    
      return sample

class Augment_hsv(object):
    def __call__(self, sample,  hgain=0.5, sgain=0.5, vgain=0.5,hsv=0.5):



      """ HSV color-space augmentation
      image merge stauration (image grayness with gain +/- 50% ) and value ( image brightness with gain +/- 50%) and Hue color with gain +/- 50 degree
      """
      if np.random.rand() < hsv:
        image, annots = sample['img'], sample['annot']
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        dtype = image.dtype
        # print(dtype)
        # print(type(dtype))
        hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.uint8))

        x = np.arange(0, 256, dtype=dtype)
        lut_hue = ((x * r[0]) % 180)
        lut_sat = np.clip(x * r[1], 0, 255)
        lut_val = np.clip(x * r[2], 0, 255)
        hue = cv2.LUT(hue, lut_hue)
        sat = cv2.LUT(sat, lut_sat)
        val = cv2.LUT(val, lut_val)        
        
        image = cv2.merge((hue, sat, val)).astype(dtype)
        image= cv2.cvtColor(image, cv2.COLOR_HSV2BGR) 
        sample = {'img': image, 'annot': annots} 

      return sample

class Augmenter_s_or_v(object):
    def __call__(self, sample,  hgain=0.0, sgain=0.0, vgain=0.0,hsv=0.5):



      """ HSV color-space augmentation
      image aug stauration (image grayness with gain +/- 50% ) Or value (image brightness with gain +/- 50%)
      
      """
      if np.random.rand() < hsv:
        if np.random.rand() < hsv:
          sgain=0.5
        else:
          vgain=0.5
        image, annots = sample['img'], sample['annot']
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        dtype = image.dtype
        # print(dtype)
        # print(type(dtype))
        hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.uint8))

        x = np.arange(0, 256, dtype=dtype)
        lut_hue = ((x * r[0]) % 180)
        lut_sat = np.clip(x * r[1], 0, 255)
        lut_val = np.clip(x * r[2], 0, 255)
        hue = cv2.LUT(hue, lut_hue)
        sat = cv2.LUT(sat, lut_sat)
        val = cv2.LUT(val, lut_val)        
        
        image = cv2.merge((hue, sat, val)).astype(dtype)
        image= cv2.cvtColor(image, cv2.COLOR_HSV2BGR) 
        sample = {'img': image, 'annot': annots}    
      return sample
class Augmenter_flip_h(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :]#slice [:, ::-1]. The array flips along the second axis.

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            sample = {'img': image, 'annot': annots}
        return sample

class Augmenter_flip_v(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            image = image[ ::-1, :]# slice [::-1]. The array flips along the first axis.

            rows, cols, channels = image.shape

            x1 = annots[:, 1].copy()
            x2 = annots[:, 3].copy()

            x_tmp = x1.copy()

            annots[:, 1] = rows - x2
            annots[:, 3] = rows - x_tmp

            sample = {'img': image, 'annot': annots}
        return sample
  

class Augmenter_RandomFlip(object):

    def __call__(self, sample, do_horizontal=False, do_vertical=False, prob=0.5):
        img,annots = sample['img'], sample['annot']
        rows, cols,_ = img.shape
        if np.random.rand() < prob:
          if np.random.rand() < prob:
            do_horizontal=True
          else:
            do_vertical=True

        if do_horizontal:
          img = img[:, ::-1, :]
          x1 = annots[:, 0].copy()
          x2 = annots[:, 2].copy()
          x_tmp = x1.copy()
          annots[:, 0] = cols - x2
          annots[:, 2] = cols - x_tmp

        elif do_vertical:
          img = img[ ::-1, :]
          x1 = annots[:, 1].copy()
          x2 = annots[:, 3].copy()
          x_tmp = x1.copy()
          annots[:, 1] = rows - x2
          annots[:, 3] = rows - x_tmp

        sample = {'img': img, 'annot': annots}
        return sample
class Augmenter_FlipHV(object):

    def __call__(self, sample, do_horizontal=True, do_vertical=True, prob=0.5):
        img,annots = sample['img'], sample['annot']
        rows, cols,_ = img.shape
        if np.random.rand() < prob:
            img = img[::-1, ::-1, :]#h[:, ::-1, :]v[ ::-1, :]hv [::-1, ::-1][::-1, ::-1, :]
            img = img[:, ::-1, :]
            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()
            x_tmp = x1.copy()
            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp
            y1 = annots[:, 1].copy()
            y2 = annots[:, 3].copy()
            y_tmp = y1.copy()
            annots[:, 1] = rows - y2
            annots[:, 3] = rows - y_tmp
            sample = {'img': img, 'annot': annots}
        return sample