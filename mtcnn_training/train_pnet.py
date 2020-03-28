import sys
sys.path.append('..')

import logging as pylog
from absl import app, flags
from absl import logging as alog
from absl.flags import FLAGS

import os
import datetime
import tensorflow as tf
from tf.config.experimental import list_physical_devices
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.callbacks import TerminateOnNaN

from modules.losses import SoftmaxLoss
from modules.utils import set_memory_growth, get_ckpt_inf
import modules.dataset as dataset

from modules.utils import load_yaml

flags.DEFINE_string('train_cfg_path', '../configs/mtcnn_pnet.yaml', 
                    'MTCNN pnet configuration file path')

flags.DEFINE_bool('log_to_file', True, 'Whether log to file')
flags.DEFINE_string('log_file', 'train_pnet', 'log filename')
flags.DEFINE_string('log_dir_path', '../logs/prepare_data', 
                    'log file directory path')

def main(_):
    if FLAGS.log_to_file: 
        alog.get_absl_handler().use_absl_log_file(FLAGS.log_file, 
                         FLAGS.log_dir_path)
    alog.set_verbosity(alog.DEBUG)
    
    cfg = load_yaml(FLAGS.train_cfg_path)
    
    # Check GPU support
    gpus = list_physical_devices('GPU')
    alog.info(f'GPUs: {gpus}')

    train_dataset = cfg['pnet_train_dataset']

    alog.info("load pnet tfrecord dataset.")
    dataset_size = cfg['num_samples']
    batch_size = cfg['batch_size']
    net_name = 'Net'
    base_lr = cfg['base_lr']
    train()



def train(net_factory, net, dataset_size, batch_size, 
          dataset_path, base_lr=0.01):
    """
    train PNet/RNet/ONet
    :param net_factory:
    :param prefix: model path
    :param end_epoch:
    :param dataset:
    :param display:
    :param base_lr:
    :return:
    """

    steps_per_epoch = dataset_size // batch_size
    alog.info(f"dataset_size: {dataset_size}, batch_size: {batch_size}, "
              + f"steps_per_epoch: {steps_per_epoch}")
    
    train_dataset = dataset.load_tfrecord_dataset(
        cfg['train_dataset'], cfg['batch_size'], cfg['binary_img'],
        is_ccrop=cfg['is_ccrop'])

    
    #label file
    label_file = os.path.join(base_dir,'train_%s_landmark.txt' % net)
    #label_file = os.path.join(base_dir,'landmark_12_few.txt')
    print(label_file)
    f = open(label_file, 'r')
    # get number of training examples
    num = len(f.readlines())
    print("Total size of the dataset is: ", num)
    print(prefix)

    #PNet use this method to get data
    if net == 'PNet':
        #dataset_dir = os.path.join(base_dir,'train_%s_ALL.tfrecord_shuffle' % net)
        dataset_dir = os.path.join(base_dir,'train_%s_landmark.tfrecord_shuffle' % net)
        print('dataset dir is:',dataset_dir)
        image_batch, label_batch, bbox_batch,landmark_batch = read_single_tfrecord(dataset_dir, config.BATCH_SIZE, net)
        
    #RNet use 3 tfrecords to get data    
    else:
        pos_dir = os.path.join(base_dir,'pos_landmark.tfrecord_shuffle')
        part_dir = os.path.join(base_dir,'part_landmark.tfrecord_shuffle')
        neg_dir = os.path.join(base_dir,'neg_landmark.tfrecord_shuffle')
        #landmark_dir = os.path.join(base_dir,'landmark_landmark.tfrecord_shuffle')
        landmark_dir = os.path.join('../../DATA/imglists/RNet','landmark_landmark.tfrecord_shuffle')
        dataset_dirs = [pos_dir,part_dir,neg_dir,landmark_dir]
        pos_radio = 1.0/6;part_radio = 1.0/6;landmark_radio=1.0/6;neg_radio=3.0/6
        pos_batch_size = int(np.ceil(config.BATCH_SIZE*pos_radio))
        assert pos_batch_size != 0,"Batch Size Error "
        part_batch_size = int(np.ceil(config.BATCH_SIZE*part_radio))
        assert part_batch_size != 0,"Batch Size Error "
        neg_batch_size = int(np.ceil(config.BATCH_SIZE*neg_radio))
        assert neg_batch_size != 0,"Batch Size Error "
        landmark_batch_size = int(np.ceil(config.BATCH_SIZE*landmark_radio))
        assert landmark_batch_size != 0,"Batch Size Error "
        batch_sizes = [pos_batch_size,part_batch_size,neg_batch_size,landmark_batch_size]
        #print('batch_size is:', batch_sizes)
        image_batch, label_batch, bbox_batch,landmark_batch = read_multi_tfrecords(dataset_dirs,batch_sizes, net)        
        
    #landmark_dir    
    if net == 'PNet':
        image_size = 12
        radio_cls_loss = 1.0;radio_bbox_loss = 0.5;radio_landmark_loss = 0.5;
    elif net == 'RNet':
        image_size = 24
        radio_cls_loss = 1.0;radio_bbox_loss = 0.5;radio_landmark_loss = 0.5;
    else:
        radio_cls_loss = 1.0;radio_bbox_loss = 0.5;radio_landmark_loss = 1;
        image_size = 48
    
    #define placeholder
    input_image = tf.placeholder(tf.float32, shape=[config.BATCH_SIZE, image_size, image_size, 3], name='input_image')
    label = tf.placeholder(tf.float32, shape=[config.BATCH_SIZE], name='label')
    bbox_target = tf.placeholder(tf.float32, shape=[config.BATCH_SIZE, 4], name='bbox_target')
    landmark_target = tf.placeholder(tf.float32,shape=[config.BATCH_SIZE,10],name='landmark_target')
    #get loss and accuracy
    input_image = image_color_distort(input_image)
    cls_loss_op,bbox_loss_op,landmark_loss_op,L2_loss_op,accuracy_op = net_factory(input_image, label, bbox_target,landmark_target,training=True)
    #train,update learning rate(3 loss)
    total_loss_op  = radio_cls_loss*cls_loss_op + radio_bbox_loss*bbox_loss_op + radio_landmark_loss*landmark_loss_op + L2_loss_op
    train_op, lr_op = train_model(base_lr,
                                  total_loss_op,
                                  num)
    # init
    init = tf.global_variables_initializer()
    sess = tf.Session()


    #save model
    saver = tf.train.Saver(max_to_keep=0)
    sess.run(init)

    #visualize some variables
    tf.summary.scalar("cls_loss",cls_loss_op)#cls_loss
    tf.summary.scalar("bbox_loss",bbox_loss_op)#bbox_loss
    tf.summary.scalar("landmark_loss",landmark_loss_op)#landmark_loss
    tf.summary.scalar("cls_accuracy",accuracy_op)#cls_acc
    tf.summary.scalar("total_loss",total_loss_op)#cls_loss, bbox loss, landmark loss and L2 loss add together
    summary_op = tf.summary.merge_all()
    logs_dir = "../logs/%s" %(net)
    if os.path.exists(logs_dir) == False:
        os.mkdir(logs_dir)
    writer = tf.summary.FileWriter(logs_dir,sess.graph)
    projector_config = projector.ProjectorConfig()
    projector.visualize_embeddings(writer,projector_config)
    #begin 
    coord = tf.train.Coordinator()
    #begin enqueue thread
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    i = 0
    #total steps
    MAX_STEP = int(num / config.BATCH_SIZE + 1) * end_epoch
    epoch = 0
    sess.graph.finalize()
    try:



        for step in range(MAX_STEP):
            i = i + 1
            if coord.should_stop():
                break
            image_batch_array, label_batch_array, bbox_batch_array,landmark_batch_array = sess.run([image_batch, label_batch, bbox_batch,landmark_batch])
            #random flip
            image_batch_array,landmark_batch_array = random_flip_images(image_batch_array,label_batch_array,landmark_batch_array)
            '''
            print('im here')
            print(image_batch_array.shape)
            print(label_batch_array.shape)
            print(bbox_batch_array.shape)
            print(landmark_batch_array.shape)
            print(label_batch_array[0])
            print(bbox_batch_array[0])
            print(landmark_batch_array[0])
            '''


            _,_,summary = sess.run([train_op, lr_op ,summary_op], feed_dict={input_image: image_batch_array, label: label_batch_array, bbox_target: bbox_batch_array,landmark_target:landmark_batch_array})

            if (step+1) % display == 0:
                #acc = accuracy(cls_pred, labels_batch)
                cls_loss, bbox_loss,landmark_loss,L2_loss,lr,acc = sess.run([cls_loss_op, bbox_loss_op,landmark_loss_op,L2_loss_op,lr_op,accuracy_op],
                                                             feed_dict={input_image: image_batch_array, label: label_batch_array, bbox_target: bbox_batch_array, landmark_target: landmark_batch_array})

                total_loss = radio_cls_loss*cls_loss + radio_bbox_loss*bbox_loss + radio_landmark_loss*landmark_loss + L2_loss
                # landmark loss: %4f,
                print("%s : Step: %d/%d, accuracy: %3f, cls loss: %4f, bbox loss: %4f,Landmark loss :%4f,L2 loss: %4f, Total Loss: %4f ,lr:%f " % (
                datetime.now(), step+1,MAX_STEP, acc, cls_loss, bbox_loss,landmark_loss, L2_loss,total_loss, lr))


            #save every two epochs
            if i * config.BATCH_SIZE > num*2:
                epoch = epoch + 1
                i = 0
                path_prefix = saver.save(sess, prefix, global_step=epoch*2)
                print('path prefix is :', path_prefix)
            writer.add_summary(summary,global_step=step)
    except tf.errors.OutOfRangeError:
        print("完成！！！")
    finally:
        coord.request_stop()
        writer.close()
    coord.join(threads)
    sess.close()
    
#
#    if cfg['train_dataset']:
#        logging.info("load ms1m dataset.")
#        dataset_len = cfg['num_samples']
#        steps_per_epoch = dataset_len // cfg['batch_size']
#        train_dataset = dataset.load_tfrecord_dataset(
#            cfg['train_dataset'], cfg['batch_size'], cfg['binary_img'],
#            is_ccrop=cfg['is_ccrop'])
#    else:
#        logging.info("load fake dataset.")
#        dataset_len = 1
#        steps_per_epoch = 1
#        train_dataset = dataset.load_fake_dataset(cfg['input_size'])
#
#    learning_rate = tf.constant(cfg['base_lr'])
#    optimizer = tf.keras.optimizers.SGD(
#        learning_rate=learning_rate, momentum=0.9, nesterov=True)
#    loss_fn = SoftmaxLoss()
#
#    ckpt_path = tf.train.latest_checkpoint('./checkpoints/' + cfg['sub_name'])
#    if ckpt_path is not None:
#        logging.info("[*] load ckpt from {}".format(ckpt_path))
#        model.load_weights(ckpt_path)
#        epochs, steps = get_ckpt_inf(ckpt_path, steps_per_epoch)
#    else:
#        logging.info("[*] training from scratch.")
#        epochs, steps = 1, 1
#
#    if FLAGS.mode == 'eager_tf':
#        # Eager mode is great for debugging
#        # Non eager graph mode is recommended for real training
#        summary_writer = tf.summary.create_file_writer(
#            './logs/' + cfg['sub_name'])
#
#        train_dataset = iter(train_dataset)
#
#        while epochs <= cfg['epochs']:
#            inputs, labels = next(train_dataset)
#
#            with tf.GradientTape() as tape:
#                logist = model(inputs, training=True)
#                reg_loss = tf.reduce_sum(model.losses)
#                pred_loss = loss_fn(labels, logist)
#                total_loss = pred_loss + reg_loss
#            
#            grads = tape.gradient(total_loss, model.trainable_variables)
#            optimizer.apply_gradients(zip(grads, model.trainable_variables))
#            
#            if tf.math.is_nan(total_loss):
#                logging.info('Error! Loss is NaN')
#                break
#
#            if steps % cfg['save_steps'] == 0:
#                logging.info('[*] save ckpt file!')
#                model.save_weights('checkpoints/{}/e_{}_b_{}.ckpt'.format(
#                    cfg['sub_name'], epochs, steps % steps_per_epoch))
#                
#                verb_str = "Epoch {}/{}: {}/{}, loss={:.2f}, lr={:.4f}"
#                logging.info(verb_str.format(epochs, cfg['epochs'],
#                                      steps % steps_per_epoch,
#                                      steps_per_epoch,
#                                      total_loss.numpy(),
#                                      learning_rate.numpy()))
#
#                with summary_writer.as_default():
#                    tf.summary.scalar(
#                        'loss/total loss', total_loss, step=steps)
#                    tf.summary.scalar(
#                        'loss/pred loss', pred_loss, step=steps)
#                    tf.summary.scalar(
#                        'loss/reg loss', reg_loss, step=steps)
#                    tf.summary.scalar(
#                        'learning rate', optimizer.lr, step=steps)
#            
#            steps += 1
#            next_epochs = steps // steps_per_epoch + 1
#            if next_epochs > epochs or next_epochs > cfg['epochs']:
#                logging.info('[*] save ckpt file for each epochs!')
#                model.save_weights('checkpoints/{}/e_{}_b_{}.ckpt'.format(
#                    cfg['sub_name'], epochs, steps % steps_per_epoch))
#            
#            epochs = next_epochs
#    else:
#        model.compile(optimizer=optimizer, loss=loss_fn,
#                      run_eagerly=(FLAGS.mode == 'eager_fit'))
#
#        mc_callback = ModelCheckpoint(
#            filepath=f'checkpoints/{cfg["sub_name"]}/' 
#                        + 'e_{epoch}_b_{batch}.ckpt',
#            save_freq=cfg['save_steps'] * cfg['batch_size'], verbose=1,
#            #save_freq='epoch', verbose=1, save_best_only=True,
#            save_weights_only=True)
#        tb_callback = TensorBoard(log_dir='logs/',
#                                  update_freq=cfg['batch_size'] * 5,
#                                  profile_batch=0)
#        tb_callback._total_batches_seen = steps
#        tb_callback._samples_seen = steps * cfg['batch_size']
#        
#        # Stop when NaN loss is detected
#        nan_callback = TerminateOnNaN()
#        
#        callbacks = [mc_callback, tb_callback, nan_callback]
#
#        history = model.fit(train_dataset,
#                            epochs=cfg['epochs'],
#                            steps_per_epoch=steps_per_epoch,
#                            callbacks=callbacks,
#                            initial_epoch=epochs - 1)

    alog.info("[*] training done!")


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
    
