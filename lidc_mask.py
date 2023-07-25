#%%
import pylidc as pl
import cv2
import numpy as np
import matplotlib.pyplot as plt

ann = pl.query(pl.Annotation).first()
print(ann.scan.patient_id)

#%%


# %%
images = ann.scan.load_all_dicom_images()
contours = sorted(ann.contours, key=lambda c: c.image_z_position)
fnames = ann.scan.sorted_dicom_file_names.split(',')
index_of_contour = [fnames.index(c.dicom_file_name) for c in contours]
min_slice = min(index_of_contour)
max_slice = max(index_of_contour)
current_slice = min_slice
img = images[current_slice].pixel_array
img = ((img - np.min(img))/(np.max(img) - np.min(img))*255).astype(np.uint8)
img = np.repeat(img[..., None], 3, -1)
mask = np.zeros_like(img)
contour = contours[0].to_matrix()[..., :2]
img = cv2.fillPoly(img, np.array([np.flip(contour)]), (255, 255, 255))
mask = cv2.fillPoly(mask, np.array([np.flip(contour)]), (255, 255, 255))
plt.imshow(mask)

# %%
