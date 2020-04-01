import tensorflow as tf

def _parse_tfrecord(binary_img=False, is_ccrop=False):
    def parse_tfrecord(tfrecord):
        if binary_img:
            features = {'image/source_id': tf.io.FixedLenFeature([], tf.int64),
                        'image/filename': tf.io.FixedLenFeature([], tf.string),
                        'image/encoded': tf.io.FixedLenFeature([], tf.string)}
            x = tf.io.parse_single_example(tfrecord, features)
            x_train = tf.image.decode_jpeg(x['image/encoded'], channels=3)
        else:
            features = {'image/source_id': tf.io.FixedLenFeature([], tf.int64),
                        'image/img_path': tf.io.FixedLenFeature([], tf.string)}
            x = tf.io.parse_single_example(tfrecord, features)
            image_encoded = tf.io.read_file(x['image/img_path'])
            x_train = tf.image.decode_jpeg(image_encoded, channels=3)

        y_train = tf.cast(x['image/source_id'], tf.float32)

        x_train = _transform_images(is_ccrop=is_ccrop)(x_train)
        y_train = _transform_targets(y_train)
        return (x_train, y_train), y_train
    return parse_tfrecord


def _transform_images(is_ccrop=False):
    def transform_images(x_train):
        x_train = tf.image.resize(x_train, (128, 128))
        x_train = tf.image.random_crop(x_train, (112, 112, 3))
        x_train = tf.image.random_flip_left_right(x_train)
        x_train = tf.image.random_saturation(x_train, 0.6, 1.4)
        x_train = tf.image.random_brightness(x_train, 0.4)
        x_train = x_train / 255
        return x_train
    return transform_images


def _transform_targets(y_train):
    return y_train


def load_tfrecord_dataset(tfrecord_name, batch_size,
                          binary_img=False, shuffle=True, buffer_size=10240,
                          is_ccrop=False):
    """load dataset from tfrecord"""
    raw_dataset = tf.data.TFRecordDataset(tfrecord_name)
    raw_dataset = raw_dataset.repeat()
    if shuffle:
        raw_dataset = raw_dataset.shuffle(buffer_size=buffer_size)
    dataset = raw_dataset.map(
        _parse_tfrecord(binary_img=binary_img, is_ccrop=is_ccrop),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


def load_fake_dataset(size):
    """load fake dataset"""
    x_train = tf.image.decode_jpeg(
        open('./data/BruceLee.JPG', 'rb').read(), channels=3)
    x_train = tf.expand_dims(x_train, axis=0)
    x_train = tf.image.resize(x_train, (size, size))

    labels = [0]
    y_train = tf.convert_to_tensor(labels, tf.float32)
    y_train = tf.expand_dims(y_train, axis=0)

    return tf.data.Dataset.from_tensor_slices((x_train, y_train))

def read_single_tfrecord(net):
    def parse_tfrecord(tfrecord):
        features={
            'image/encoded': tf.FixedLenFeature([], tf.string),#one image  one record
            'image/label': tf.FixedLenFeature([], tf.int64),
            'image/roi': tf.FixedLenFeature([4], tf.float32),
            'image/landmark': tf.FixedLenFeature([10],tf.float32)
        }
        
        parsed_tfrecord = tf.io.parse_single_example(tfrecord, features)
        if net == 'PNet':
            image_size = 12
        elif net == 'RNet':
            image_size = 24
        elif net == "ONet":
            image_size = 48
        else:
            raise Exception(f"Invalid net name: {net}")
    
        image = tf.decode_raw(parsed_tfrecord['image/encoded'], tf.uint8)
        image = tf.reshape(image, [image_size, image_size, 3])
        image = (tf.cast(image, tf.float32)-127.5) / 128
        
        # image = tf.image.per_image_standardization(image)
        label = tf.cast(parsed_tfrecord['image/label'], tf.float32)
        roi = tf.cast(parsed_tfrecord['image/roi'],tf.float32)
        landmark = tf.cast(parsed_tfrecord['image/landmark'],tf.float32)
        
        return image, label, roi, landmark

    return parse_tfrecord

def load_train_dataset(tfrecords_filename, batch_size, net_name):
    dataset = tf.data.TFRecordDataset(tfrecords_filename)
    dataset = dataset.shuffle().batch(batch_size, drop_remainder=True)
    
    image, label, roi, landmark = dataset.map(read_single_tfrecord(net_name))
        
    return image, label, roi, landmark