# Hybrid-Images
Description:
Hybrid images are static images that change in interpretation as a function of the viewing distance. The basic idea is that high frequency tends to dominate perception when it is available, but, at a distance, only the low frequency (smooth) part of the signal can be seen. By blending the high frequency portion of one image with the low-frequency portion of another, you get a hybrid image that leads to different interpretations at different distances.


How to Run:

To run this, you first have to get two images that you want to hybridize in .png form and put these two images in the images folder in the project.

Then you call:

python hybrid_image_starter.py [first_image] [second_image]

(For example, say you have two images called a.png and b.png

To run it, you type in:
python hybrid_image_starter.py a b )

Then a small window should output and you need to select two points per image. The points basically dictate the alignment.

After you've selected two points per image, the final hybridized image should pop up in a new window.
