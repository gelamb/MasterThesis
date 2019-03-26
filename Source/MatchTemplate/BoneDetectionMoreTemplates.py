import cv2
import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def find_bones(input_image, template_path, output_path, i):

    methods = ['cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF_NORMED']

    # operations on the full image
    img2 = cv2.imread(input_image, 0)
    image_to_crop = Image.open(input_image)
    z, q = img2.shape[::-1]

    j = 0
    for t in os.listdir(template_path):
        if t.endswith(".jpg"):
            full_t = template_path + t
            template = cv2.imread(full_t, 0)
            # create a flipped template
            template_flipped = cv2.flip(template, 1) 
            # get shapes of original images
            w, h = template.shape[::-1]  # means that template is w rows and h columns
     
            for meth in methods:
                img = img2.copy()
                method = eval(meth)

                # Apply template Matching
                res = cv2.matchTemplate(img, template, method)
                res_flipped = cv2.matchTemplate(img, template_flipped, method)

                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                min_val_f, max_val_f, min_loc_f, max_loc_f = cv2.minMaxLoc(res_flipped)
                print("Comparing image {} with template {}, min_val is {} and max_val is {} and min_val_f {} and max_val_f {}"
                      .format(input_image, t, min_val, max_val, min_val_f, max_val_f))

                # print step by step squares
                bottom_right = (min_loc[0] + w, min_loc[1] + h)  # somma base e altezza del template
                bottom_right_f = (min_loc_f[0] + w, min_loc_f[1] + h)

                cv2.rectangle(img, min_loc, bottom_right, (255, 255, 255), thickness=30)
                cv2.rectangle(img, min_loc_f, bottom_right_f, (0, 0, 255), thickness=30)

                plt.imshow(img, cmap='gray')
                plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
                plt.suptitle(meth)
                plt.show()

                # select the max and min position of the max and min value compared the three templates images
                if j == 0:
                    min_global = min_val
                    min_global_loc = min_loc
                    max_global = max_val
                    max_global_loc = max_loc

                    min_global_f = min_val_f
                    min_global_loc_f = min_loc_f
                    max_global_f = max_val_f
                    max_global_loc_f = max_loc_f
                else:
                    if (min_val < min_global):
                        min_global = min_val
                        min_global_loc = min_loc
                    if (max_val > max_global):
                        max_global = max_val
                        max_global_loc = max_loc

                    if (min_val_f < min_global_f):
                        min_global_f = min_val_f
                        min_global_loc_f = min_loc_f
                    if (max_val_f > max_global_f):
                        max_global_f = max_val_f
                        max_global_loc_f = max_loc_f

                j += 1

    print("max value is now {} and min value {}".format(max_global, min_global))
    # print chosen square

    bottom_right = (min_global_loc[0] + w, min_global_loc[1] + h)  # somma base e altezza del template
    bottom_right_f = (min_global_loc_f[0] + w, min_global_loc_f[1] + h)

    cv2.rectangle(img, min_global_loc, bottom_right, (255, 255, 255), thickness=30)
    cv2.rectangle(img, min_global_loc_f, bottom_right_f, (0, 0, 255), thickness=30)

    plt.imshow(img, cmap='gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)
    plt.show()

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_global_loc
        top_left_f = min_global_loc_f
    else:
        top_left = max_global_loc
        top_left_f = max_global_loc_f
            
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

    crop_image.save(output_path + 'Cropped{}.jpg'.format(i))
    crop_image_f.save(output_path + 'Cropped_flipped{}.jpg'.format(i))



if __name__ == "__main__":

    path = "/Users/leonardotanzi/Desktop/TesiMagistrale/Audisio/ImmaginiBacinoCopiaCrop/"
    template_path = "/Users/leonardotanzi/Desktop/MasterThesis/Templates/"
    output_path = "/Users/leonardotanzi/Desktop/MasterThesis/Output/"
    i = 0

    for f in os.listdir(path):
        if f.endswith(".jpg"):
            full_f = path + f
            find_bones(full_f, template_path, output_path, i)
            i += 1
            if i == 5:
                break
