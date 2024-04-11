#!/usr/bin/env python

__author__ = "Lorenz A. Kapsner"
__copyright__ = "Copyright 2023, Universitätsklinikum Erlangen"

from skimage.draw import ellipse
import numpy as np
from matplotlib import pyplot as plt
from skimage import exposure, filters
import logging

from scipy import signal, stats
from sklearn import mixture
from scipy.cluster import vq

def remove_background_noise(proc_img):
    # compute background
    compute_cutoff_area = proc_img.copy()[:25,:]

    # subtract 95%-pct quantile of noise
    proc_img = proc_img - np.quantile(compute_cutoff_area, .95)
    # again compute noise and set everything below abs(mean of noise) to 0
    compute_cutoff_area2 = proc_img.copy()[:25,:]
    proc_img[proc_img <= abs(compute_cutoff_area2.mean())] = 0
    
    return proc_img

def thorax_masker(
        img,
        detection_window: int = 10,
        debug=False
    ):

    # sternum detection

    # save shape
    img_shape = img.shape
    # plt.imshow(img)
    # plt.show()
    
    # x-min and x-max of detection-window
    x_min = int((img_shape[1] / 2) - (detection_window / 2))
    x_max = int(x_min + detection_window)

    # plt.imshow(img)

    # get detection window
    proc_img = img.copy()[:, x_min:x_max]
    # plt.imshow(proc_img)
    # plt.show()

    # remove noise
    proc_img = remove_background_noise(proc_img)

    # https://www.sciencedirect.com/science/article/pii/S2405844018327178?ref=pdf_download&fr=RR-9&rr=7d4714287ed018cd#br0380
    # wiener filter, noise reduction
    wf = signal.wiener(
        im=proc_img.astype("float32"),
        mysize=5,
        noise=None
    )
    # plt.imshow(wf)
    # plt.show()

    n_clusters = 4

    X = wf.reshape((-1, 1))
    cluster_centers, cluster_labels = vq.kmeans2(
        data=X,
        k=n_clusters,
        iter=10,
        minit="++",
        seed=123
    )
    kmc = cluster_centers.squeeze()[cluster_labels].reshape(wf.shape)
    # plt.imshow(kmc)
    # plt.show()

    # binary mask
    cluster_centers = cluster_centers.squeeze()
    binary_mask_kmc = kmc > cluster_centers[stats.rankdata(cluster_centers) == 1]
    # plt.imshow(binary_mask_kmc)
    # plt.show()

    # gaussian mixture
    gm = mixture.GaussianMixture(
        n_components=n_clusters,
        covariance_type="full",
        init_params="k-means++",
        random_state=123
    )
    gm_res = gm.fit_predict(X=proc_img.reshape((-1, 1)))
    
    # get frequency of cluster values
    clus_counts = np.bincount(gm_res)
    # rank values according to frequency in array in decreasing order
    cluster_ranks = stats.rankdata(clus_counts)
    cluster_px_replace = abs(cluster_ranks - len(cluster_ranks))
    gm_res_remap = gm_res.copy()
    for _pxval in np.unique(gm_res):
        gm_res_remap[gm_res == _pxval] = int(cluster_px_replace[_pxval])
    # quality check
    if gm_res_remap[0] != 0:
        logging.info("Remapping values.")
        gm_res_remap2 = gm_res_remap.copy()
        # replace first value with 0
        gm_res_remap2[gm_res_remap == gm_res_remap[0]] = 0
        
        # compute difference values that need to be resetted now
        diff_vals = set(np.unique(gm_res_remap)).difference({gm_res_remap[0]})
        diff_vals_ranks = stats.rankdata(list(diff_vals))
        for _i, _remapval in enumerate(diff_vals):
            gm_res_remap2[gm_res_remap == _remapval] = int(diff_vals_ranks[_i])
        gm_res_remap = gm_res_remap2
    gmc = gm_res_remap.squeeze().reshape(proc_img.shape)
    # plt.imshow(gmc)
    # plt.show()
    binary_mask_gmc = gmc > 1
    # plt.imshow(binary_mask_gmc)
    # plt.show()

    sternum_position = int(np.argwhere(binary_mask_kmc.sum(axis=1) > 0.7 * detection_window)[0])
    
    sternum_position_gmc = int(np.argwhere(binary_mask_gmc.sum(axis=1) > 0.7 * detection_window)[0])

    if abs(sternum_position - sternum_position_gmc) > 5:
        if sternum_position_gmc > sternum_position and abs(sternum_position - sternum_position_gmc) <= 15:
            logging.info("Using sternum-position from 'gmm-clustering'")
            # weighted average
            sternum_position = int((sternum_position_gmc * 3 + sternum_position) / 4)
        
    # sternum plus 5%
    sternum_cutoff = int(sternum_position + 0.05 * img_shape[0])
    # plt.imshow(center_cols)
    # plt.axhline(y=sternum_position, color="g")
    # plt.axhline(y=sternum_cutoff, color="r")

    # img_square_masked
    img_square_masked = img.copy()
    img_square_masked[sternum_cutoff:,:] = img.min()
    #plt.imshow(img_square_masked)

    # img_ellipsis_masked
    img_ellipsis_masked = img.copy()
    rr, cc = ellipse(
        r=img_shape[0],
        c=(img_shape[1] / 2),
        r_radius=img_shape[0] - sternum_cutoff,
        c_radius=img_shape[1],
        shape=img_shape
    )
    img_ellipsis_masked[rr, cc] = img.min()
    #plt.imshow(img_ellipsis_masked)

    if debug:
        return img_square_masked, img_ellipsis_masked, kmc, binary_mask_kmc, sternum_position, sternum_cutoff
    else:
        return img_square_masked, img_ellipsis_masked
