# Description

Python script for cropping clear borders of any color in images with some degree of noise tolerance.

More information is available in [previous version's readme](https://github.com/WerctFourth/python-border-autocrop)

# Differences

* Uses pyvips/libvips instead of Pillow for 16-bit/band and more format support
* Resizes images using Magic Kernel Sharp (a=6, v=7), Numpy + Numba resizer implementation
* Saves 16-bit/band images after resizing (8-bit input images without resizing will be saved as 8-bit)
* Settings are saved in JSON with an ability to load custom settings files
* Can save and load job JSONs with custom settings for any file
* Saves to PNG (up to 16 bit), JXL (internal libvips, up to 16 bit), AVIF (still external, because internal ilbvips encoder doesn't support 8+ bit images; up to 12 bit)
* Option to ignore already vertically cropped space in horizontal crop

# Requirements
Python, [pyvips](https://pypi.org/project/pyvips/), [numpy](https://pypi.org/project/numpy/), [numba](https://pypi.org/project/numba/). 

Tested on Python 3.13.1, pyvips 2.2.3, Numpy 2.1.3, Numba 0.61.0.

Requires external native [libvips library](https://github.com/libvips/libvips/releases).

# Resources

* [Magic Kernel official site](https://johncostella.com/magic/)
* [Magic Kernel Rust implementation](https://lib.rs/crates/magic-kernel)
* [pica JS resizer](https://github.com/nodeca/pica)