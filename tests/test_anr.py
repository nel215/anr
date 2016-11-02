# coding:utf-8
import numpy as np
from skimage.transform import rescale
from skimage.measure import compare_psnr
from anr import ANR, extract_from_imgs, clip


def test_extract():
    n, h, w = 10, 128, 64
    ps = (5, 5)
    imgs = [np.random.rand(h, w) for _ in range(n)]
    features, patches = extract_from_imgs(imgs, ps)
    assert patches.shape == (n * (h - 4) * (w - 4), 5 * 5)
    assert features.shape == (n * (h - 4) * (w - 4), 5 * 5 * 5)


def test_fit():
    np.random.seed(0)
    base = np.random.rand(32, 32)
    imgs = [clip(base + np.random.randn(32, 32) / 100) for _ in range(10)]
    a = ANR()
    a.fit(imgs)

    test_img = base
    res_img = a.upscale(rescale(test_img, 0.5))
    base_line = rescale(rescale(test_img, 0.5), 2.0)
    assert compare_psnr(test_img, res_img) > compare_psnr(test_img, base_line)
