import random
import tensorflow as tf


def get_band(x, min_band_size, max_band_size, band_type, mask):
    assert band_type.lower() in ['freq', 'time'], f"band_type must be in ['freq', 'time']"
    if band_type.lower() == 'freq':
        axis = 2
    else:
        axis = 1
    band_size =  random.randint(min_band_size, max_band_size)
    mask_start = random.randint(0, tf.shape(x)[axis] - band_size) 
    mask_end = mask_start + band_size
    mask = tf.Variable(mask)
    if band_type.lower() == 'freq':
        values = tf.ones((mask.shape[0], band_size))
        mask[:, mask_start:mask_end].assign(values) 
    if band_type.lower() == 'time':
        values = tf.ones((band_size, mask.shape[1]))
        mask[mask_start:mask_end, :].assign(values)
    return mask

def specmix(x, y, prob, min_band_size, max_band_size, max_frequency_bands=3, max_time_bands=3):
    if prob < 0:
        raise ValueError('prob must be a positive value')

    k = random.random()
    if k > 1 - prob:
        indices = tf.range(start=0, limit=tf.shape(x)[0], dtype=tf.int32)
        shuffled_indices = tf.random.shuffle(indices)
        mask = tf.zeros(tf.shape(x)[1:3])
        num_frequency_bands = random.randint(1, max_frequency_bands)
        for i in range(1, num_frequency_bands):
            mask = get_band(x, min_band_size, max_band_size, 'freq', mask)
        num_time_bands = random.randint(1, max_time_bands)
        for i in range(1, num_time_bands):
            mask = get_band(x, min_band_size, max_band_size, 'time', mask)
        total_pixels = tf.cast(tf.shape(x)[1] * tf.shape(x)[2], dtype=tf.float32)
        lam = tf.math.reduce_sum(mask)  / total_pixels
        x = x * (1 - mask) + tf.gather(x, shuffled_indices) * mask
        y = y * (1 - lam) + tf.gather(y, shuffled_indices) * (lam)
        return x, y
    else:
        return x, y