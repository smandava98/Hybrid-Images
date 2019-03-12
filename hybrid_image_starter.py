import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from align_image_code import align_images
import numpy as np
import cv2
import skimage
import scipy
import pickle


#RUN THIS WITH TWO IMAGES IN .png FORMAT
def generate_fft_plot(im):
    im = skimage.color.rgb2grey(im)
    plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(im)))), cmap='jet')  
    plt.show()  

def generate_greyscale(im):
    im = skimage.color.rgb2grey(im)
    plt.imshow(im, cmap='gray')
    plt.show()

def generate_rgb(im):
    plt.imshow(im)
    plt.show()

def generate_filter_plot(im):
    plt.pcolor(im, cmap='jet')
    plt.show()

def hybrid_image(im1, im2, sigma1, sigma2, lam=0.8, plot_filters=False, plot_fft=False):
    # Rule of thumb from slides: set filter half-width to 3 sigma
    width_1 = sigma1 * 3    
    width_2 = sigma2 * 3

    x_filter_1 = np.zeros((2 * width_1 + 1, 2 * width_1 + 1))
    for i in range(len(x_filter_1)):
        x_filter_1[i] = np.arange(-width_1, width_1 + 1, step=1)
    y_filter_1 = -x_filter_1.T
    
    filter_1 = np.exp(-(x_filter_1 ** 2 + y_filter_1 ** 2) / (2 * sigma1 ** 2))
    filter_1 = filter_1 / np.sum(filter_1)

    x_filter_2 = np.zeros((2 * width_2 + 1, 2 * width_2 + 1))
    for i in range(len(x_filter_2)):
        x_filter_2[i] = np.arange(-width_2, width_2 + 1, step=1)
    y_filter_2 = -x_filter_2.T
    
    filter_2 = np.exp(-(x_filter_2 ** 2 + y_filter_2 ** 2) / (2 * sigma2 ** 2))
    filter_2 = filter_2 / np.sum(filter_2)

    low_pass_1 = cv2.filter2D(im1, -1, filter_1)
    low_pass_2 = cv2.filter2D(im1, -1, filter_2)
    high_pass_1 = im1 - low_pass_1
    high_pass_2 = im2 - low_pass_2

    hybrid = lam * low_pass_1 + (1 - lam) * high_pass_2

    if plot_filters:
        deltas = np.zeros((2 * width_1 + 1, 2 * width_1 + 1))
        deltas[width_1 + 1, width_1] = 1.0
        hpf_1 = deltas - filter_1

        generate_filter_plot(filter_1)
        generate_filter_plot(hpf_1)
    if plot_fft:
        for im in (low_pass_1, high_pass_2, hybrid):
            generate_fft_plot(im)
    return hybrid

def show_scales(im, gray=True):    
    y, x = im.shape[0], im.shape[1]
    
    im2 = cv2.resize(im, dsize=(0, 0), fx=0.5, fy=0.5)
    im3 = cv2.resize(im2, dsize=(0, 0), fx=0.5, fy=0.5)
    im4 = cv2.resize(im3, dsize=(0, 0), fx=0.5, fy=0.5)

    new_image = np.zeros((y, round(x * 1.5), im.shape[2]))
    new_image[:y, :x, :] = im
    new_image[:im2.shape[0],x:x + im2.shape[1], :] = im2
    new_image[im2.shape[0]:im2.shape[0]+ im3.shape[0], round(5 * x / 4):round(5 * x / 4) + im3.shape[1],:] = im3
    new_image[round(5 * y / 8):round(5 * y/8) + im4.shape[0], round(9 * x / 8):round(9 * x / 8) + im4.shape[1],:] = im4
   
    if gray:
        generate_greyscale(new_image)
    else:
        generate_rgb(new_image)
    

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('im_names', nargs='*', type=str)
    parser.add_argument('--show_originals', action='store_true')
    parser.add_argument('--plot_filters', action='store_true')
    parser.add_argument('--plot_fft', action='store_true')
    parser.add_argument('--gray', action='store_true')
    parser.add_argument('--restore', action='store_true')
    parser.add_argument('--s1', type=int, default=6)
    parser.add_argument('--s2', type=int, default=6)
    parser.add_argument('--lam', type=float, default=0.8)
    args = parser.parse_args()

    assert(len(args.im_names) == 2)

    if args.restore:
        save_file = 'pickle/{}_{}.pkl'.format(args.im_names[0], args.im_names[1])
        with open(save_file, 'rb') as f:
            hybrid = pickle.load(f)
    else:

        im1 = plt.imread('images/{}.png'.format(args.im_names[0]))
        im1 = im1[:,:,:3] #only keeps 3 channels -rgb

        im2 = plt.imread('images/{}.png'.format(args.im_names[1]))
        im2 = im2[:,:,:3]

        im1_aligned, im2_aligned = align_images(im1, im2)

        if args.show_originals:
            generate_greyscale(im1_aligned)
            generate_greyscale(im1_aligned)
            generate_fft_plot(im1_aligned)
            generate_fft_plot(im2_aligned)

        sigma1 = args.s1
        sigma2 = args.s2
        save_file = 'pickle/{}_{}.pkl'.format(args.im_names[0], args.im_names[1])
        hybrid = hybrid_image(im1_aligned, im2_aligned, sigma1, sigma2, lam=args.lam, plot_filters=args.plot_filters, plot_fft=args.plot_fft)

        with open(save_file, 'wb') as f:
            pickle.dump(hybrid, f)

    if args.gray:
        generate_greyscale(hybrid)
    else:
        generate_rgb(hybrid)
    
    show_scales(hybrid, gray=args.gray)

if __name__ == "__main__":
    main()



