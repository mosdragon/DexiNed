"""
Load in data partitions using the TF Record files.
"""
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE
BUFFER_SIZE = 1024
BATCH_SIZE = 16

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
    tf_trn = read_tfrecord_dataset("datasets/biped_trn.tfrecord")
    tf_val = read_tfrecord_dataset("datasets/biped_val.tfrecord")
    tf_tst = read_tfrecord_dataset("datasets/biped_tst.tfrecord")

    # Create the training and testing data loaders
    train = tf_trn.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val = tf_val.map(load_image)
    test = tf_tst.map(load_image)

    train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    val_dataset = val.batch(BATCH_SIZE)
    test_dataset = test.batch(BATCH_SIZE)

    return (train_dataset, val_dataset, test_dataset)
