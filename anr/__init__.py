# coding:utf-8
import sys
import numpy as np
from scipy import ndimage
from scipy.linalg import inv
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA, MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d
from skimage.transform import rescale


def clip(img):
    img = np.minimum(np.ones(img.shape), img)
    img = np.maximum(np.zeros(img.shape), img)
    return img


def filter_img(img):
    filters = [
        np.array([[-1, 0, 1]]),
        np.array([[-1, 0, 1]]).T,
        np.array([[1, 0, -2, 0, 1]]),
        np.array([[1, 0, -2, 0, 1]]).T,
    ]

    h, w = img.shape
    res = np.zeros((h, w, 5))
    res[:, :, 0] = img
    for i, f in enumerate(filters):
        res[:, :, i + 1] = ndimage.convolve(img, f, mode='constant')
    return res


def extract_from_imgs(imgs, ps):
    features = []
    patches = []
    for hr_img in imgs:
        mr_img = rescale(rescale(hr_img, 0.5), 2.0)
        f = extract_patches_2d(filter_img(mr_img), ps)
        features.append(f)

        diff = hr_img - mr_img
        p = extract_patches_2d(diff, ps)
        patches.append(p)

        sys.stdout.write('.')
        sys.stdout.flush()
    print()
    features = np.concatenate(features)
    patches = np.concatenate(patches)
    n = features.shape[0]

    return features.reshape(n, -1), patches.reshape(n, -1)


class ANR(object):

    def __init__(self):
        self.pca = None
        self.lr_dict = None
        self.ps = (9, 9)

    def fit(self, imgs):
        features, patches = extract_from_imgs(imgs, self.ps)
        print('features size:', features.shape)
        print('patches size:', patches.shape)

        # dim. reduction
        print("start pca")
        features -= np.mean(features, axis=0)
        pca = PCA(n_components=0.999)
        features_pca = pca.fit_transform(features)
        self.pca = pca
        print('features_pca size:', features_pca.shape)

        # lr dict. learning
        print("start dictionary learning")
        dict_size = 128
        dico = MiniBatchDictionaryLearning(n_components=dict_size)
        code = dico.fit_transform(features_pca)
        self.lr_dict = dico.components_

        # construct hr dict.
        # hr_dict = inv(code.T.dot(code)).dot(code.T).dot(patches)
        hr_dict = inv(
            code.T.dot(code) + 0.01 * np.eye(code.shape[1])).dot(
                code.T).dot(patches)
        # print(patches - code.dot(hr_dict))

        # create anchors
        print("start creating anchors")
        dist_mat = cdist(self.lr_dict, self.lr_dict)
        neighbor_size = 8
        kth_mat = np.partition(
            dist_mat, neighbor_size, 1)[:, neighbor_size:neighbor_size+1]
        anchors = dist_mat < kth_mat
        self.projections = []
        for anchor in anchors:
            ln = self.lr_dict[anchor].T
            hn = hr_dict[anchor].T
            p = hn.dot(
                inv(ln.T.dot(ln) + 0.01 * np.eye(ln.shape[1])).dot(ln.T))
            self.projections += [p]

    def upscale(self, lr_img):
        mr_img = rescale(lr_img, 2)
        hr_img = np.zeros(mr_img.shape)
        weight = np.zeros(mr_img.shape)
        h, w = mr_img.shape
        f = extract_patches_2d(filter_img(mr_img), self.ps)
        f = f.reshape(f.shape[0], -1)
        f -= np.mean(f, axis=0)
        f_pca = self.pca.transform(f)
        f_cnt = 0
        anchor_idx = np.argmin(cdist(f_pca, self.lr_dict), axis=1)
        py, px = self.ps
        for y in range(h-py+1):
            for x in range(w-px+1):
                proj = self.projections[anchor_idx[f_cnt]]
                p = proj.dot(f_pca[f_cnt].T)
                hr_img[y:y+py, x:x+px] += p.reshape(self.ps)
                weight[y:y+py, x:x+px] += 1
                f_cnt += 1
        hr_img /= weight
        hr_img += mr_img
        hr_img = clip(hr_img)

        return hr_img
