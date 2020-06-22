"""
Load in data partitions using the TF Record files.
"""
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE
SHUFFLE_SIZE = 48960  # Same size as training set
PREFETCH_SIZE = 2
BATCH_SIZE = 8

def parse_record(example_proto):
    # Create a dictionary describing the features.
    image_feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'segmentation_mask': tf.io.FixedLenFeature([], tf.string),
    }
    # Parse the input tf.Example proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, image_feature_description)


def record_to_images(record):
    features = parse_record(record)
    image = tf.io.decode_jpeg(features['image'], channels=3)
    annotation = tf.io.decode_png(features['segmentation_mask'], channels=1)
    return (image, annotation)


def read_tfrecord_dataset(filepath):
    records = tf.data.TFRecordDataset(filepath)
    return records.map(record_to_images)


def normalize(image, mask):
    image = tf.cast(image, tf.float32) / 255.0
    mask = tf.cast(mask, tf.float32) / 255.0
    return (image, mask)

@tf.function
def load_image(img, mask):
    TARGET_W = TARGET_H = 400
    resized_image = tf.image.resize(img, (TARGET_H, TARGET_W))
    resized_mask = tf.image.resize(mask, (TARGET_H, TARGET_W))
    image, mask = normalize(resized_image, resized_mask)
    return image, mask


def get_loaders():
    # train = read_tfrecord_dataset("datasets/biped_trn.tfrecord")
    train = read_tfrecord_dataset("datasets/biped_val.tfrecord")
    val = read_tfrecord_dataset("datasets/biped_val.tfrecord")
    test = read_tfrecord_dataset("datasets/biped_tst.tfrecord")

    # Create the training and testing data loaders
    train = train.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val = val.map(load_image)
    test = test.map(load_image)

    train = train.cache('./tmp/train.data')
    train = train.shuffle(SHUFFLE_SIZE, reshuffle_each_iteration=True)
    train = train.batch(BATCH_SIZE).prefetch(buffer_size=PREFETCH_SIZE)

    val = val.cache("./tmp/val.data")
    val = val.batch(BATCH_SIZE).prefetch(buffer_size=PREFETCH_SIZE)
    test = test.batch(BATCH_SIZE)

    return (train, val, test)
