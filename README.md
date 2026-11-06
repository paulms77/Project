# Project — Small Projects Collection

Four self-contained projects from coursework and personal experiments: two
classical computer-vision pipelines built on OpenCV, one deep-learning
experiment, and one game.

Each is a single file and stands alone.

| Project | What it does | Source |
| --- | --- | --- |
| **Car licence plate recognition** | Locates and extracts the plate region from a vehicle photo using contour geometry | [`CarLicenseDetector.py`](CarLicenseDetector.py) |
| **Retinal vessel detection** | Segments blood vessels in fundus photographs with morphological filtering | [`EyesDetector.py`](EyesDetector.py) |
| **Lottery number prediction** | LSTM over historical draw data — an experiment in why this cannot work | [`LottoProject_paul77ms.ipynb`](LottoProject_paul77ms.ipynb) |
| **snake.io** | A Snake clone in Pygame | [`Snake_io-update.ipynb`](Snake_io-update.ipynb) |

---

## Car licence plate recognition

Finds the plate without any machine learning, purely from the geometry of the
characters on it.

Contours are extracted from the thresholded image, then filtered on size and
shape:

```python
MIN_AREA,  MAX_AREA   = 80, 550
MIN_WIDTH, MIN_HEIGHT = 2, 7
MIN_RATIO, MAX_RATIO  = 0.2, 2.3
```

The useful part is `find_chars`, which then looks for contours that behave like
*a row of characters* rather than like isolated blobs — candidates are grouped
when they are close together relative to their diagonal, aligned in angle, and
similar in area, width and height:

```python
MAX_DIAG_LENGTH = 4      # distance, in multiples of the contour diagonal
MAX_ANGLE_DIFF  = 9.0    # degrees off horizontal
MAX_AREA_DIFF   = 0.5
MAX_WIDTH_DIFF  = 0.9
MAX_HEIGHT_DIFF = 0.3
MIN_N_MATCHED   = 3      # at least 3 characters to count as a plate
```

The function recurses over the unmatched contours, so several plates in one
image are each found independently.

```bash
pip install opencv-python numpy
# expects images in ./CarLicensePlate/
python CarLicenseDetector.py
```

## Retinal vessel detection

Extracts the vascular tree from fundus images using the same colour-channel idea
as lane detection — pick the channel with the best contrast, then isolate
structures by shape.

The green channel carries the most vessel contrast, so it is taken alone,
CLAHE-equalised, and run through morphological opening/closing at three
increasing kernel sizes (5×5, 11×11, 23×23). Subtracting that background estimate
from the equalised image leaves the thin vessel structures behind.

Accuracy is scored against ground-truth masks:

```python
def getAccuracy(detected, gt):
    test  = cv2.bitwise_and(detected, gt)
    return cv2.countNonZero(test) / cv2.countNonZero(gt)
```

```bash
pip install opencv-python numpy
# expects images in ./Eyes/images/ and masks in ./Eyes/result/
python EyesDetector.py
```

## Lottery number prediction

Korean lottery draws encoded as 45-dimensional one-hot vectors, fed to an LSTM
that outputs a probability per number, with draws sampled from that
distribution.

It is an honest experiment with a foregone conclusion: draws are independent and
uniformly random, so there is no signal to learn and the network converges to
approximately the marginal frequency of each number. Worth keeping as a
demonstration of what a model does when asked to predict noise — the loss curve
looks fine, and the output is still worthless.

```bash
pip install torch pandas numpy
jupyter notebook LottoProject_paul77ms.ipynb
```

## snake.io

A Snake clone built with Pygame.

```bash
pip install pygame
jupyter notebook Snake_io-update.ipynb
```
