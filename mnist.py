import gzip
import os
import numpy
from six.moves import urllib
import tensorflow as tf
from tensorflow.python.framework import dtypes
from collections import namedtuple

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10


Datasets = namedtuple('Datasets', ['train', 'validation', 'test'])


class DataSet(object):
    def __init__(self, images, labels, dtype=dtypes.float32,
                 fake_data=False, reshape=False):
        """Construct a DataSet. """
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                            dtype)
        if fake_data:
            self._num_examples = 10000
        else:
            assert images.shape[0] == labels.shape[0], (
                'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
            self._num_examples = images.shape[0]

            # Convert shape from [num examples, rows, columns, depth]
            # to [num examples, rows*columns] (assuming depth == 1)
            if reshape:
                assert images.shape[3] == 1
                images = images.reshape(images.shape[0],
                                        images.shape[1] * images.shape[2])
            if dtype == dtypes.float32:
                # Convert from [0, 255] -> [0.0, 1.0].
                images = images.astype(numpy.float32)
                images = numpy.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._reshape = reshape

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
            if self._reshape:
                fake_images = numpy.full((batch_size, IMAGE_SIZE**2),
                                         0.5, dtype=numpy.float32)
            else:
                fake_images = numpy.full((batch_size, IMAGE_SIZE, IMAGE_SIZE, 1),
                                         0.5, dtype=numpy.float32)
            fake_labels = numpy.zeros((batch_size,), dtype=numpy.int32)
            return fake_images, fake_labels
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


def _maybe_download(filename, train_dir='data'):
    """Download the data from Yann's website, unless it's already here."""
    if not tf.gfile.Exists(train_dir):
        tf.gfile.MakeDirs(train_dir)
    filepath = os.path.join(train_dir, filename)
    if not tf.gfile.Exists(filepath):
        filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
        with tf.gfile.GFile(filepath) as f:
            size = f.size()
        print('Successfully downloaded', filename, size, 'bytes.')
    return filepath


def _read32(bytestream):
    dt = numpy.dtype(numpy.uint32).newbyteorder('>')
    return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]


def _extract_images(filename):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth].
    Args:
      filename of gzip file

    Returns:
      Values are rescaled from [0, 255] down to [-0.5, 0.5].

      data: A 4D unit8 numpy array [index, y, x, depth].

    Raises:
      ValueError: If the bytestream does not start with 2051.

    """
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                             (magic, filename))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = numpy.frombuffer(buf, dtype=numpy.uint8)
        data = data.reshape((num_images, rows, cols, 1))
        return data


def _extract_labels(filename):
    """Extract the labels into a vector of int64 label IDs."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                             (magic, filename))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = numpy.frombuffer(buf, dtype=numpy.uint8)
        return labels


def read_mnist(train_dir, fake_data=False, reshape=False, validation_size=5000, ):
    if fake_data:
        def fake():
            return DataSet([], [], fake_data=True, reshape=reshape)

        train = fake()
        validation = fake()
        test = fake()
        return Datasets(train=train, validation=validation, test=test)

    TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
    TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
    TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
    TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

    local_file = _maybe_download(TRAIN_IMAGES, train_dir)
    train_images = _extract_images(local_file)

    local_file = _maybe_download(TRAIN_LABELS, train_dir)
    train_labels = _extract_labels(local_file)

    local_file = _maybe_download(TEST_IMAGES, train_dir)
    test_images = _extract_images(local_file)

    local_file = _maybe_download(TEST_LABELS, train_dir)
    test_labels = _extract_labels(local_file)

    if not 0 <= validation_size <= len(train_images):
        raise ValueError(
            'Validation size should be between 0 and {}. Received: {}.'
            .format(len(train_images), validation_size))

    validation_images = train_images[:validation_size]
    validation_labels = train_labels[:validation_size]
    train_images = train_images[validation_size:]
    train_labels = train_labels[validation_size:]

    train = DataSet(train_images, train_labels, reshape=reshape)
    validation = DataSet(validation_images, validation_labels, reshape=reshape)
    test = DataSet(test_images, test_labels, reshape=reshape)

    return Datasets(train=train, validation=validation, test=test)