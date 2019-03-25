import cv2
import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt



def find_bones(inputImage, templateImage, i):
    
    
    img2 = cv2.imread(inputImage, 0)
    image_to_crop = Image.open(inputImage)
    #img2 = img.copy()
    template = cv2.imread(templateImage, 0)
    #create a flipped template
    template_flipped = cv2.flip(template, 1) 
    #get shapes of original images
    w, h = template.shape[::-1] #means that template is w rows and h columns
    z, q = img2.shape[::-1]

    # All the 6 methods for comparison in a list
    methods = [#'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR_NORMED', 
                'cv2.TM_SQDIFF_NORMED']

    for meth in methods:
        img = img2.copy()
        method = eval(meth)

        # Apply template Matching
        res = cv2.matchTemplate(img, template, method)
        res_flipped= cv2.matchTemplate(img, template_flipped, method)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        min_val_f, max_val_f, min_loc_f, max_loc_f = cv2.minMaxLoc(res_flipped)

        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
            top_left_f = min_loc_f
        else:
            top_left = max_loc
            top_left_f = max_loc_f
        
        bottom_right = (top_left[0] + w, top_left[1] + h) #somma base e altezza del template
        bottom_right_f = (top_left_f[0] + w, top_left_f[1] + h)

        if(top_left[1] + h) * 2 > q:
            b_r = q
        else:
            b_r = (top_left[1] + h) * 2


        if(top_left_f[1] + h) * 2 > q:
            b_r_f = q
        else:
            b_r_f = (top_left_f[1] + h) * 2
        
        box = (top_left[0], top_left[1], top_left[0] + w, b_r)
        box_f = (top_left_f[0], top_left_f[1], top_left_f[0] + w, b_r_f)

        crop_image = image_to_crop.crop(box)
        crop_image_f = image_to_crop.crop(box_f)

        crop_image.save('Cropped{}.jpg'.format(i))
        crop_image_f.save('Cropped_flipped{}.jpg'.format(i))

        '''
        cv2.rectangle(img, top_left, bottom_right, (255, 255, 255), thickness = 30)
        cv2.rectangle(img, top_left_f, bottom_right_f, (255, 255, 255), thickness = 30)

        plt.imshow(img, cmap = 'gray')
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        plt.suptitle(meth)
        plt.show()
        '''

if __name__ == "__main__":

    path = "/Users/leonardotanzi/Desktop/TesiMagistrale/Audisio/ImmaginiBacinoCopiaCrop/"
    templateImage = "Bacino.jpg"
    i = 0

    for f in os.listdir(path):
        if f.endswith(".jpg"):
            full_f = "/Users/leonardotanzi/Desktop/TesiMagistrale/Audisio/ImmaginiBacinoCopiaCrop/" + f
            find_bones(full_f, templateImage, i)
            i += 1
            if i == 5:
                break

