"""Main script. Contain model definition and training code."""
import os

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D, Input, MaxPool2D
from tensorflow.keras.layers import Flatten, Reshape
from tensorflow.keras.layers import Dense, Permute, Softmax, PReLU
from tensorflow.keras.models import Model


class NetWork(object):

    def __init__(self, inputs, mode='train'):

        self.inputs = inputs
        self.mode = mode
        self.out_put = []
        # Default model which is also current training model
        self.net_model = None
        # Current training task index
        self.cur_task_index = -1

        if self.mode == 'train':
            self.tasks = [inp[0] for inp in inputs]
            self.net_model = self.setup(self.mode)
            self.cur_task_index = 0
        else:
            self.net_model = self.setup(self.mode)

    def setup(self, task='data'):
        raise NotImplementedError('Must be implemented by the subclass.')

    def load(self, weights_path):
        if os.path.exists(weights_path):
            self.net_model.load_weights_file(weights_path, by_name=True)

    def get_all_output(self):
        return self.out_put

    def validate_padding(self, padding):
        assert padding in ('SAME', 'VALID')


class PNet(NetWork):

    def create_pnet(self, input_shape=(12, 12, 3), network_mode='train'):
        p_inp = Input(input_shape)
    
        p_layer = Conv2D(10, kernel_size=(3, 3), strides=(1, 1), 
                         padding="valid", name='conv1')(p_inp)
        p_layer = PReLU(shared_axes=[1, 2], name='PReLU1')(p_layer)
        p_layer = MaxPool2D(pool_size=(2, 2), strides=(2, 2), 
                            padding="same")(p_layer)
    
        p_layer = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), 
                         padding="valid", name='conv2')(p_layer)
        p_layer = PReLU(shared_axes=[1, 2], name='PReLU2')(p_layer)
    
        p_layer = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), 
                         padding="valid", name='conv3')(p_layer)
        p_layer = PReLU(shared_axes=[1, 2], name='PReLU3')(p_layer)
    
        if network_mode == 'train':
            p_classifier = Conv2D(2, kernel_size=(1, 1), strides=(1, 1), 
                              activation='softmax', name='cls1')(p_layer)
            p_classifier = Reshape((2,))(p_classifier)
            p_bbox_regress = Conv2D(4, kernel_size=(1, 1), strides=(1, 1), 
                              name='bbox1')(p_layer)
            p_bbox_regress = Reshape((4,))(p_bbox_regress)
            p_net = Model([p_inp], [p_classifier, p_bbox_regress])
        else:
            p_classifier = Conv2D(2, kernel_size=(1, 1), strides=(1, 1), 
                                  activation='softmax', name='cls1')(p_layer)
            p_classifier = Reshape((2,))(p_classifier)
                
            p_bbox_regress = Conv2D(4, kernel_size=(1, 1), strides=(1, 1), 
                                  name='bbox1')(p_layer)
            p_bbox_regress = Reshape((4,))(p_bbox_regress)
        
            p_net = Model([p_inp], [p_classifier, p_bbox_regress])
       
        return p_net
    
    def setup(self, network_mode='train'):
        return self.create_pnet(network_mode=network_mode)


def read_and_decode(serialized_data, label_type, shape):
    context, sequence = tf.parse_single_example(
        serialized_data,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label_raw': tf.FixedLenFeature([], tf.string),
        })
    image = tf.decode_raw(context['image_raw'], tf.uint8)
    image = tf.cast(image, tf.float32)

    image = (image - 127.5) * (1. / 128.0)
    image.set_shape([shape * shape * 3])
    image = tf.reshape(image, [shape, shape, 3])
    label = tf.decode_raw(context['label_raw'], tf.float32)

    if label_type == 'cls':
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        label.set_shape([2])
    elif label_type == 'bbx':
        label.set_shape([4])
    elif label_type == 'pts':
        label.set_shape([10])

    return image, label


def inputs(tfrecords_filename, batch_size, num_epochs, label_type, shape):
    capacity=1000 + 3 * batch_size
    
    images, labels = tf.data.TFRecordDataset(tfrecords_filename) \
                    .shuffle(capacity) \
                    .batch(batch_size) \
                    .repeat(num_epochs) \
                    .map(lambda x: read_and_decode(x, label_type, shape))

    return images, labels


def train_net(Net, training_data, base_lr, loss_weight,
              train_mode, num_epochs=[1, None, None],
              batch_size=64, load_model=False, load_filename=None,
              save_model=False, save_filename=None,
              num_iter_to_save=10000):

    images = []
    labels = []
    tasks = ['cls', 'bbx', 'pts']
    shape = 12
    if Net.__name == 'PNet':
        shape = 12
        train_mode = 2
    elif Net.__name__ == 'RNet':
        shape = 24
        train_mode = 2
    elif Net.__name__ == 'ONet':
        shape = 48
        train_mode = 3
    else:
        raise Exception('Invalid training net model')
    
    for index in range(train_mode):
        image, label = inputs(filename=[training_data[index]],
                              batch_size=batch_size,
                              num_epochs=num_epochs[index],
                              label_type=tasks[index],
                              shape=shape)
        images.append(image)
        labels.append(label)
    
    
    if train_mode == 2:
        net = Net((('cls', images[0]), ('bbx', images[1])))
    elif train_mode == 3:
        net = Net((('cls', images[0]), ('bbx', images[1]), ('pts', images[2])))
    else:
        raise Exception(f'Invalid train_mode {train_mode}')

    out_put = net.get_all_output()
    cls_output = tf.reshape(out_put[0], [-1, 2])
    bbx_output = tf.reshape(out_put[1], [-1, 4])
    pts_output = tf.reshape(out_put[2], [-1, 10])

    # cls loss
    softmax_loss = loss_weight[0] * \
        tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=labels[0],
                                                logits=cls_output))
    weight_losses_cls = net.get_weight_decay()['cls']
    losses_cls = softmax_loss + tf.add_n(weight_losses_cls)

    # bbx loss
    square_bbx_loss = loss_weight[1] * \
        tf.reduce_mean(tf.squared_difference(bbx_output, labels[1]))
    weight_losses_bbx = net.get_weight_decay()['bbx']
    losses_bbx = square_bbx_loss + tf.add_n(weight_losses_bbx)

    # pts loss
    square_pts_loss = loss_weight[2] * \
        tf.reduce_mean(tf.squared_difference(pts_output, labels[2]))
    weight_losses_pts = net.get_weight_decay()['pts']
    losses_pts = square_pts_loss + tf.add_n(weight_losses_pts)

    global_step_cls = tf.Variable(1, name='global_step_cls', trainable=False)
    global_step_bbx = tf.Variable(1, name='global_step_bbx', trainable=False)
    global_step_pts = tf.Variable(1, name='global_step_pts', trainable=False)

    train_cls = tf.train.AdamOptimizer(learning_rate=base_lr) \
                        .minimize(losses_cls, global_step=global_step_cls)
    train_bbx = tf.train.AdamOptimizer(learning_rate=base_lr) \
                        .minimize(losses_bbx, global_step=global_step_bbx)
    train_pts = tf.train.AdamOptimizer(learning_rate=base_lr) \
                        .minimize(losses_pts, global_step=global_step_pts)

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
    config.gpu_options.allow_growth = True

    loss_agg_cls = [0]
    loss_agg_bbx = [0]
    loss_agg_pts = [0]
    step_value = [1, 1, 1]

    with tf.Session(config=config) as sess:
        sess.run(init_op)
        saver = tf.train.Saver(max_to_keep=200000)
        if load_model:
            saver.restore(sess, load_filename)
        else:
            net.load(load_filename, sess, prefix)
        if save_model:
            save_dir = os.path.split(save_filename)[0]
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            while not coord.should_stop():
                choic = np.random.randint(0, train_mode)
                if choic == 0:
                    _, loss_value_cls, step_value[0] = sess.run(
                        [train_cls, softmax_loss, global_step_cls])
                    loss_agg_cls.append(loss_value_cls)
                elif choic == 1:
                    _, loss_value_bbx, step_value[1] = sess.run(
                        [train_bbx, square_bbx_loss, global_step_bbx])
                    loss_agg_bbx.append(loss_value_bbx)
                else:
                    _, loss_value_pts, step_value[2] = sess.run(
                        [train_pts, square_pts_loss, global_step_pts])
                    loss_agg_pts.append(loss_value_pts)

                if sum(step_value) % (100 * train_mode) == 0:
                    agg_cls = sum(loss_agg_cls) / len(loss_agg_cls)
                    agg_bbx = sum(loss_agg_bbx) / len(loss_agg_bbx)
                    agg_pts = sum(loss_agg_pts) / len(loss_agg_pts)
                    print(
                        'Step %d for cls: loss = %.5f' %
                        (step_value[0], agg_cls), end='. ')
                    print(
                        'Step %d for bbx: loss = %.5f' %
                        (step_value[1], agg_bbx), end='. ')
                    print(
                        'Step %d for pts: loss = %.5f' %
                        (step_value[2], agg_pts))
                    loss_agg_cls = [0]
                    loss_agg_bbx = [0]
                    loss_agg_pts = [0]

                if save_model and (step_value[0] % num_iter_to_save == 0):
                    saver.save(sess, save_filename, global_step=step_value[0])

        except tf.errors.OutOfRangeError:
            print(
                'Done training for %d epochs, %d steps.' %
                (num_epochs[0], step_value[0]))
        finally:
            coord.request_stop()

        coord.join(threads)
