# Description

Python script for cropping clear borders of any color in images with some degree of noise tolerance.

More information is available in [previous version's readme](https://github.com/WerctFourth/python-border-autocrop)

# Differences

* Uses pyvips/libvips instead of Pillow for 16-bit/band and more format support
* Resizes images using Magic Kernel Sharp (a=6, v=7), Numpy + Numba resizer implementation
* Detects grayscale images saved as RGB and converts them back to grayscale (pyvips + numpy implementation)
* Saves 16-bit/band images after resizing (8-bit input images without resizing will be saved as 8-bit)
* Settings are saved in JSON with an ability to load custom settings files
* Can save and load job JSONs with custom settings for any file
* Saves to PNG (up to 16 bit), JXL (internal libvips, up to 16 bit), AVIF (still external, because internal ilbvips encoder doesn't support 8+ bit images; up to 12 bit)
* Option to ignore already vertically cropped space in horizontal crop
* Quality approximation for JXL, AVIF and HEIF using SSIMULACRA2 as an evaluation metric

# Requirements
Python, [pyvips](https://pypi.org/project/pyvips/), [numpy](https://pypi.org/project/numpy/), [numba](https://pypi.org/project/numba/). 

Optional: [pyvips-binary](https://pypi.org/project/pyvips-binary/). The script can use it, but it's limited in regards to supported formats, for example JXL.

Tested on Python 3.14.2, pyvips 3.0.0, pyvips-binary 8.16.1, Numpy 2.2.5, Numba 0.61.2.

Can use external native [libvips library](https://github.com/libvips/libvips/releases). It's recommended to use that instead of pyvips-binary.

# Quality approximation details
Here I talk a bit about quality approximation algorithm in this script. Essentially it saves a PNG source image after all previous processing steps (crop \ resize). Then it encodes it at a certain quality level. Then the encoded image is being decoded back to PNG and evaluated against the source PNG using SSIMULACRA2 metric. If the score is good enough, then this particular encoded image is saved, and all the temporary files are deleted. If not, then the encoding quality value is adjusted accordingly and the process repeats.

The algorithm is quite fast at that point, in my opinion (~2.5 iters for AVIF and ~2.1 for JXL for ±0.5 SSIM tolerance, but it depends heavily on image types and their consistency, so your mileage may vary).

## In short
For AVIF\HEIF: Bisection algorithm + initial point adjustment + jump list + duplicate elimination + near SSIM difference check + quantizer\quality conversion + some more exit conditions just for sanity check.

For JXL: Secant algorithm + initial point adjustment + jump list + dup elimination + several exit conditions.

## Bisection algorithm
Bisection algorithm is used as a basis for AVIF\HEIF quality approximation. It's not used for JXL because in AVIF quality steps are quite large (only 64 of them essentially), but in JXL steps are much more granular (essentially 0.0-25.0 in float32 AFAIK). This makes bisection in JXL slower to converge that secant.

## Initial point adjustment
For both bisection and secant, the initial point doesn't necessarily need to be placed in the center of initial boundaries. It can be placed anywhere. For the first several encoded files it's placed according to a setting in the JSON file. Then the algorithm takes the resulting quality values (up to 100) and takes median value of them and sets the initial point according to it.

It stores 100 of previous resulting quality values in order to not expand the quality value list too much on large file batches, and for any next file to be able to make a significant change to the initial point (in the case of a drastic change in content, for example from drawn images to photos to computer graphics).

It enables the algorithm to one-shot the prediction in some cases, noticeably shortening the encoding time.

## Jump list
The algorithm remembers the SSIM and quality values of the first iteration. Then, after the successful approximation, it saves the rounded to integer SSIM value (from the first iter) and the number of encoder quality steps it took from the first to the last iteration for this particular file. It helps to make more meaningful encoder quality decisions on second and subsequent approximation iters. 

If the particular SSIM value wasn't found in the dictionary, the algorithm falls back to pure bisection (or divides the first quality value by 2 in the case of JXL). 

So on the second and later iters, instead of relying purely on bisection, the algorithm checks if the particular SSIM was already approximated in the past, and how many quality steps it took to get the desired SSIM value, and uses that. It stores up to 100 quality values for each integer SSIM and gets median value of them.

When the jump list is empty, it naturally doesn't help to get any speed boost. So there exists a warm-up phase in the beginning of processing when the algorithm relies on pure bisection to populate the values to use in the future. The processing can be quite slow at this point. To mitigate that, it stores a jump cache JSON with the values from previous runs besides the script file and loads it on every startup.

For example:
```
Warm-up phase:
Iter 1: Q 30, SSIM 76
...
...
Iter 6: Q 36, SSIM 80
Jump value of 6 is stored for SSIM 76

Jump list-enabled phase:
Iter 1: Q 28, SSIM 76
Jump value 6 is found for SSIM 76
Iter 2: Q 34, SSIM 80
```

## Duplicate elimination
It's simple: if it just so happens that the algorithm already checked SSIM for a particular encoder quality and stumbles upon that quality again, it skips the encoding process on that particular step and continues as if nothing happened. It helps to save time.

The algorithm does that by keeping encoded files, SSIMs and encoder quality values of all the previous iterations of one particular file.

## Near SSIM difference check
The algorithm constantly checks for previous iterations with the quality difference of 1. If it detects that one SSIM value is below the threshold and the next one is above it, then it stops early because there is nothing to improve in terms of accuracy. For example, there were 2 iters with q=35 and SSIM 78 and q=36 and SSIM 81. They both aren't in the tolerance bracket, but the algorithm can't squeeze one more quality level between them to approximate, so it stops and chooses the closest one to the target SSIM level (q=36 SSIM 81 in this example for target SSIM 80).

This feature works best with initial point adjustment and jump lists. If the first approximation is near the target SSIM value, but not specifically there, the jump list usually contains the jump value of 1. It helps to narrow down the SSIM on just two iterations in many cases.

## Quantizer \ quality conversion
AVIF encoder (at least, its command-line version which I use) has JPEG-like quality metric (0-100%). But, as far as I understand, the encoder just maps it internally to a quantizer value (which is only 0-63). This causes the situations when nearby quality settings may result in identical files (for example, 35 and 36). So this script internally uses the 0-62 quality scale (-1 because 0 is for lossless, which is useless in this particular case) and expands it back to 0-100 to feed it to the encoder. This helps to avoid unnecessary iterations.

## Exit conditions
There are several exit conditions in the algorithm:
* The SSIM value in within the tolerance bounds
* Large SSIM difference in the nearby quality values (see "Near SSIM difference check")
* Iteration count reached 10 (JXL-only, for edge cases)
* Very high SSIM (above the tolerance bound) and very low quality setting (usually means empty image)
* Current iter encoding quality equals the previous iter quality (To prevent infinite cycling)
* Current iter SSIM value = previous iter SSIM value (for JXL, because for secant to work they can't be equal)
* Skipped (quality duplicates) iters count reached 5 (for JXL, safety measure to prevent secant infinite cycling in edge cases)

## SSIM tolerance bounds
SSIM tolerance is implemented to shorten the approximation time without heavily decreasing the accuracy (but you can decrease it further if you want, the setting it there). It's absolutely possible to set it to 0, at least for AVIF, the algorithm accounts for that (see "Near SSIM difference check"), but it will be slightly slower. Of course that won't result in 100% perfect SSIM approximation due to large AVIF quality steps.

The default value is ±0.5 which is in my opinion good enough. In my experiments, the approximation accuracy hasn't been higher than ±0.42 SSIM points on average, so 0.5 is probably the safe value to leave.

## Secant algorithm
Secant algorithm is used to approximate quality for Jpeg XL due to its small quality steps (0.0 - 25.0 float). This means, in its slowest form, the algorithm has to use 3+ iterations to get something useful. But with the help of initial point adjustment (at first iter) and jump list (at second iter only) it can produce meaningful results even before the secant formula kicks in.

At the first iter it uses either the quality value from the settings or the iteratively adjusted quality value. On the second iter, it just divides the first quality value in half if it's not in the jump list yet. If it is, the algorithm uses its median value, just like in bisection. Jump list is being used only at the second iter. If both failed, then at the third and subsequent iters it uses the secant formula.

## Failed \ inefficient approaches
Some ideas that have proven largely useless.

### SSIM biasing
The idea was to perform the approximation on a very fast encoding speed setting (for example AVIF s=9), and then do the final encoding on a very slow speed (AVIF s=3 or even =1). It was back in the time when this algorithm was very slow (4 or more iters on average if I recall correctly). So in short, it didn't work reliably due to SSIM differences for different speed settings on a particular file (for example SSIM 80 for s=9 may become SSIM 83 at s=3). Accounting (essentially, averaging the difference) for that didn't really work because the difference isn't some constant value and changes with every image, which tanks the approximation accuracy.

The code for that function is still there, but it's useless, deprecated and may be removed completely in the future.

### Bound shortening
For AVIF, in bisection, if at the first iter the SSIM metric is near the target SSIM (say, within 10 points), then limit the encoding quality prediction interval to 10 from one side. This largely worked, but became obsolete after implementing the jump list. Also, it can somewhat worsen the prediction accuracy for some edge cases (for example, largely empty image). It's no longer in use.

### Mixing bisection and secant
I tried using bisection for the first 2 iterations and the secant formula for the third for AVIF. Long story short, it didn't produce any positive results. Secant is just useless for AVIF.

### Static training data
I haven't tried to implement this myself, but I don't think this would work quite well. Because I believe there is no one-size-fits-all solution. Some formula derived from a mainly photo dataset may work good on photos, but then slow the approximation down on CGs or drawn pictures, for example. And then what, use multiple formulas? How to detect what's on the image quickly and reliably in this case? Manually? So instead of relying on a fixed dataset I implemented jump lists which are adjusted automatically at runtime for a particular image set which is being encoded.

## Future ideas
Some ideas that would be good to be implemented.

### External metric
The idea is to use some external metric to influence the choice of the initial quality point (at the first iter) to one-shot the approximation more frequently and save on iterations. I haven't found the appropriate and reliable metric to use yet (for example, image entropy didn't yield any meaningful results).

# Resources

* [Magic Kernel official site](https://johncostella.com/magic/)
* [Magic Kernel Rust implementation](https://lib.rs/crates/magic-kernel)
* [pica JS resizer](https://github.com/nodeca/pica)
* [Oavif](https://github.com/gianni-rosato/oavif) 