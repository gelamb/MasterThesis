import cv2
import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

img1 = cv2.imread("/Users/leonardotanzi/Desktop/TesiMagistrale/Audisio/ImmaginiBacinoCopiaCrop/kitchen.jpg", 0)          # queryImage
img2 = cv2.imread("/Users/leonardotanzi/Desktop/MasterThesis/Templates/fork.jpg", 0) # trainImage


# Initiate ORB detector
orb = cv2.ORB_create()

# find the keypoints with ORB
kp1 = orb.detect(img1, None)
kp2 = orb.detect(img2, None)
# compute the descriptors with ORB
kp1, des1 = orb.compute(img1, kp1)
kp2, des2 = orb.compute(img2, kp2)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# Match descriptors.
matches = bf.match(des1, des2)
# Sort them in the order of their distance.
matches = sorted(matches, key=lambda x: x.distance)
# Draw first 10 matches.
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)
plt.imshow(img3)
plt.show()


'''
def find_bones(input_image, template_name, output_path, i):

    methods = [  # 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR_NORMED',
               'cv2.TM_SQDIFF_NORMED']

    # operations on the full image
    img2 = cv2.imread(input_image, 0)
    image_to_crop = Image.open(input_image)
    z, q = img2.shape[::-1]

    template = cv2.imread(template_name, 0)
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
              .format(input_image, template_name, min_val, max_val, min_val_f, max_val_f))

        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
            top_left_f = min_loc_f
        else:
            top_left = max_loc
            top_left_f = max_loc_f

        bottom_right = (top_left[0] + w, top_left[1] + h)  # somma base e altezza del template
        bottom_right_f = (top_left_f[0] + w, top_left_f[1] + h)

        if (top_left[1] + h) * 2 > q:
            b_r = q
        else:
            b_r = (top_left[1] + h) * 2

        if (top_left_f[1] + h) * 2 > q:
            b_r_f = q
        else:
            b_r_f = (top_left_f[1] + h) * 2

        box = (top_left[0], top_left[1], top_left[0] + w, b_r)
        box_f = (top_left_f[0], top_left_f[1], top_left_f[0] + w, b_r_f)

        crop_image = image_to_crop.crop(box)
        crop_image_f = image_to_crop.crop(box_f)

        crop_image.save('Cropped{}.jpg'.format(i))
        crop_image_f.save('Cropped_flipped{}.jpg'.format(i))


        cv2.rectangle(img, top_left, bottom_right, (255, 255, 255), thickness = 30)
        cv2.rectangle(img, top_left_f, bottom_right_f, (0, 0, 0), thickness = 30)

        plt.imshow(img, cmap = 'gray')
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        plt.suptitle(meth)
        plt.show()


if __name__ == "__main__":

    path = "/Users/leonardotanzi/Desktop/TesiMagistrale/Audisio/ImmaginiBacinoCopiaCrop/"
    template_path = "/Users/leonardotanzi/Desktop/MasterThesis/Templates/Bacino_left.jpg"
    output_path = "/Users/leonardotanzi/Desktop/MasterThesis/Output/"
    i = 0

    for f in os.listdir(path):
        if f.endswith(".jpg"):
            full_f = path + f
            find_bones(full_f, template_path, output_path, i)
            i += 1
            if i == 10:
                break
'''