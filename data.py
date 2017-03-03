import random
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


def generate_spots(min_count, max_count):
    count = random.randint(min_count, max_count)
    flags = np.array([1.0] * count + [0.0] * (max_count - count)) # 1.0 - spot is visible
    data = np.random.random((count, 2)) # random coordinates for visible points
    data = np.vstack([data, np.zeros(((max_count - count), 2))]) # zero coordinates for invisible points
    data = np.hstack([flags.reshape((max_count, 1)), data])
    data = data[np.lexsort((data[:, 0], data[:, 1]))] # sort by X coord; invisible first
    return data


def spots_to_map(spots, map_width, map_height):
    spots_map = np.zeros((map_height, map_width))
    for spot_idx in xrange(spots.shape[0]):
        if spots[spot_idx, 0] < 0.5: # invisible
            continue
        spots_x = int(map_width * spots[spot_idx, 1])
        spots_y = int(map_height * spots[spot_idx, 2])
        spots_map[spots_y, spots_x] = 1.0
    return spots_map


def spots_to_picture(spots, pic_width, pic_height, rotate_angle=None):
    # image with some noise
    im = np.random.random((pic_height, pic_width)) / 20
    im = Image.fromarray(np.uint8(im * 255))
    draw = ImageDraw.Draw(im)

    # draw markings
    max_margin = pic_height / 8
    left_margin = random.randint(1, max_margin)
    top_margin = random.randint(1, max_margin)
    right_margin = random.randint(1, max_margin)
    bottom_margin = random.randint(1, max_margin)
    draw.rectangle([left_margin, top_margin, pic_width - right_margin, pic_height - bottom_margin], outline=64)
    draw.line([pic_width / 2, top_margin, pic_width / 2, pic_height - bottom_margin], fill=64)

    # draw spots as circls of random radius
    for spot_idx in xrange(spots.shape[0]):
        if spots[spot_idx, 0] < 0.5:  # invisible
            continue
        # radius = random.randint(1, (pic_width + pic_height) / 16)
        radius = 1
        spots_x = left_margin + int((pic_width - left_margin - right_margin) * spots[spot_idx, 1])
        spots_y = top_margin + int((pic_height - top_margin - bottom_margin) * spots[spot_idx, 2])
        draw.ellipse((spots_x - radius, spots_y - radius, spots_x + radius, spots_y + radius), fill=128, outline=128)

    # random rotation
    if rotate_angle is not None:
        im = im.rotate(random.randint(-rotate_angle, rotate_angle))

    return np.asarray(im) / 255.0
