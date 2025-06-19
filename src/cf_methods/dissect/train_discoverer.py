# The difference from train_explainer.py is that we introduce a
# regularizer to tease apart which knob shifted.

import sys
import os
import os.path as osp
import warnings
import numpy as np
from tqdm import tqdm 
import tf_slim as slim
import tensorflow as tf
# import tensorflow.compat.v1 as tf
from tensorflow.keras.utils import to_categorical 


from explainer.ops import safe_log
# from explainer.networks_128 import Discriminator_Ordinal as Discriminator_Ordinal_128
# from explainer.networks_128 import Generator_Encoder_Decoder as Generator_Encoder_Decoder_128
# from explainer.networks_128 import Discriminator_Contrastive as Discriminator_Contrastive_128

from explainer.networks_64 import Discriminator_Ordinal 
from explainer.networks_64 import Generator_Encoder_Decoder
from explainer.networks_64 import Discriminator_Contrastive

from src.utils import get_config
from src.datasets.dataset_builder import DatasetBuilder
from src.cf_methods.dissect.utils import save_images, convert_ordinal_to_binary
from src.cf_methods.dissect.losses import discriminator_loss, generator_loss, l1_loss, contrastive_regularizer_loss

# tf.disable_v2_behavior()
warnings.filterwarnings("ignore", category=DeprecationWarning)



class DiscovererTrainer:
    def __init__(self, config):
        self.config = config
        self.BATCH_SIZE = config.batch_size
        self.EPOCHS = config.epochs
        self.channels = config.num_channels
        self.input_size = config.data.img_size
        self.NUM_CLASSES = config.data.num_classes
        self.NUM_BINS = config.num_bins
        self.factual_class = config.factual_class
        self.target_class = config.target_class
        self.lambda_GAN = config.lambda_GAN
        self.lambda_cyc = config.lambda_cyc
        self.lambda_cls = config.lambda_cls
        self.save_summary = config.save_summary # number of epochs after which to save summary
        self.save_ckpt = config.save_ckpt # number of epochs after which to save checkpoint
        self.k_dim = config.k_dim
        self.lambda_r = config.lambda_r
        self.disentangle = self.k_dim > 1
        self.discriminate_every_nth = config.discriminate_every_nth
        self.generate_every_nth = config.generate_every_nth

        self._create_directories()
        self._load_pretrained_classifier()

    def _create_directories(self):
        # Create directories for logs, checkpoints, samples, and tests
        assets_dir = osp.join(self.config['log_dir'], self.config['name'])
        self.log_dir = osp.join(assets_dir, 'log')
        self.sample_dir = osp.join(assets_dir, 'sample')
        self.test_dir = osp.join(assets_dir, 'test')

        self.ckpt_dir_continue = self.config.ckpt_dir_continue
        if  self.ckpt_dir_continue == '':
            self.continue_train = False
            self.ckpt_dir = osp.join(assets_dir, 'checkpoints')
        else:
            self.continue_train = True
            self.ckpt_dir = self.ckpt_dir_continue

        # make directory if not exist
        try:
            os.makedirs(self.log_dir, exist_ok=True)
            os.makedirs(self.ckpt_dir, exist_ok=True)
            os.makedirs(self.sample_dir, exist_ok=True)
            os.makedirs(self.test_dir, exist_ok=True)
        except Exception as e:
            print("Error in creating directories: ", e)

    def _load_pretrained_classifier(self):
        model_weights = self.config.classifier_weights
        self.pretrained_classifier = tf.keras.models.load_model(model_weights)
        print("Pre-trained classifier loaded from: ", model_weights)
    
    # Initialize networs
    def initialize_networks(self):
        self.G = Generator_Encoder_Decoder(name="generator")
        self.D = Discriminator_Ordinal(name="discriminator")
        self.R = Discriminator_Contrastive(name="disentangler")
    
    def initialize_optimizers(self):
        self.G_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0., beta_2=0.9)
        self.D_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0., beta_2=0.9)
        self.R_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0., beta_2=0.9)
    
    @tf.function
    def train_step(self, x_source, y_s, y_t, y_reg, y_r=None, train_phase=True):
        y_source = y_s[:, 0] # legacy:  y_s[:, 0]
        y_target = y_t[:, 0] # legacy:  y_t[:, 0]
        
        if self.disentangle:
            y_r_0 = tf.zeros_like(y_r, name='y_r_0')

        # Training step with GradientTape
        print("Calculating losses...")
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape, tf.GradientTape() as reg_tape:
            # Generator forward pass
            if self.disentangle:
                fake_target_img, _ = self.G(
                    x_source, y_reg * self.NUM_BINS + y_target,            
                    self.NUM_BINS * self.k_dim) # training=train_phase   #fake_target_img_embedding
                fake_source_img, _ = self.G(
                    fake_target_img, y_reg * self.NUM_BINS + y_source, 
                    self.NUM_BINS * self.k_dim)  # _ -> fake_source_img_embedding
                fake_source_recons_img, _ = self.G(
                    x_source, y_reg * self.NUM_BINS + y_source, 
                    self.NUM_BINS * self.k_dim)  # x_source_img_embedding
                print(" Generator forward pass completed.")

            # Discriminator forward pass
            real_source_logits = self.D(x_source, y_s, self.NUM_BINS, "NO_OPS")
            fake_target_logits = self.D(fake_target_img, y_t, self.NUM_BINS, None)
            print(" Discriminator forward pass completed.")

            # Pre-trained classifier
            real_img_prediction = self.pretrained_classifier(x_source, training=False)
            # real_img_prediction = tf.argmax(real_img_logit, axis=1)

            fake_img_prediction = self.pretrained_classifier(fake_target_img, training=False)
            # fake_img_prediction = tf.argmax(fake_img_logit, axis=1)

            real_img_recons_prediction = self.pretrained_classifier(fake_source_img, training=False)
            # real_img_recons_prediction = tf.argmax(real_img_recons_logit, axis=1)

            # Classifier evaluation losses
            real_p = tf.cast(y_target, tf.float32) * 1.0 / float(self.NUM_CLASSES - 1)
            fake_q = fake_img_prediction[:, self.target_class]
            fake_evaluation = (real_p * safe_log(fake_q)) + ((1 - real_p) * safe_log(1 - fake_q))
            fake_evaluation = -tf.reduce_mean(fake_evaluation)
            
            # TODO: check the label of the reconstruction command! Perhaps it should be of the factual class? 
            recons_evaluation = (real_img_prediction[:, self.target_class] * safe_log(real_img_recons_prediction[:, self.target_class])) + \
                    ((1 - real_img_prediction[:, self.target_class]) * safe_log(1 - real_img_recons_prediction[:, self.target_class]))
            recons_evaluation = -tf.reduce_mean(recons_evaluation) 
            print(" Classifier's predictions are obtained.")

            # Regularizer losses (if disentangling)
            if self.disentangle:
                reg_fake_target_v_source_logits = self.R(tf.concat([x_source, fake_target_img], 
                                                                   axis=-1), self.k_dim)
                reg_fake_source_v_target_logits = self.R(tf.concat([fake_target_img, fake_source_img], 
                                                                   axis=-1), self.k_dim)
                reg_fake_source_v_source_logits = self.R(tf.concat([x_source, fake_source_img],
                                                                    axis=-1), self.k_dim)
                reg_fake_source_recon_v_source_logits = self.R(tf.concat([x_source, fake_source_recons_img],
                                                                          axis=-1), self.k_dim)

                R_fake_target_v_source_loss, R_fake_target_v_source_acc = contrastive_regularizer_loss(reg_fake_target_v_source_logits, y_r)
                R_fake_source_v_target_loss, R_fake_source_v_target_acc = contrastive_regularizer_loss(reg_fake_source_v_target_logits, y_r)

                R_fake_source_v_source_loss, R_fake_source_v_source_acc = contrastive_regularizer_loss(reg_fake_source_v_source_logits, y_r_0)
                R_fake_source_recon_v_source_loss, R_fake_source_recon_v_source_acc = contrastive_regularizer_loss(reg_fake_source_recon_v_source_logits, y_r_0)

                R_loss = R_fake_target_v_source_loss + R_fake_source_v_target_loss + R_fake_source_v_source_loss + R_fake_source_recon_v_source_loss
                R_loss = R_loss * self.lambda_r
                print("Regularizer losses calculated.")
            
            # Calculate losses
            D_loss_GAN, D_acc, D_precision, D_recall = discriminator_loss('hinge', real_source_logits, fake_target_logits)
            D_loss = D_loss_GAN * self.lambda_GAN
            print("Discriminator losses calculated.")
            
            G_loss_GAN = generator_loss('hinge', fake_target_logits)
            G_loss_cyc = l1_loss(x_source, fake_source_img)
            G_loss_rec = l1_loss(x_source, fake_source_recons_img)

            # recons_evaluation * self.lambda_cls is abscent in the formula (7)
            if self.disentangle:
                G_loss = G_loss_GAN * self.lambda_GAN + G_loss_rec * self.lambda_cyc \
                    + G_loss_cyc * self.lambda_cyc + fake_evaluation * self.lambda_cls \
                    + recons_evaluation * self.lambda_cls + R_loss * self.lambda_r
                print("Generator losses calculated with disentanglement.")

        # Calculate gradients
        discriminator_gradients = disc_tape.gradient(D_loss, self.D.trainable_variables)
        if self.disentangle:
            generator_gradients = gen_tape.gradient(G_loss, self.G.trainable_variables + self.R.trainable_variables)
            regularizer_gradients = reg_tape.gradient(R_loss, self.regularizer.trainable_variables)
        else:
            generator_gradients = gen_tape.gradient(G_loss, self.G.trainable_variables)
        
        # Apply gradients
        self.d_optimizer.apply_gradients(zip(discriminator_gradients, self.D.trainable_variables))
        self.g_optimizer.apply_gradients(zip(generator_gradients, 
                                        self.G.trainable_variables + (self.R.trainable_variables if self.disentangle else [])))
        if self.disentangle:
            self.r_optimizer.apply_gradients(zip(regularizer_gradients, self.R.trainable_variables))
        
        # Update metrics
        self.train_loss.update_state(G_loss)

        # Return losses for logging
        results = {
            'G_loss': G_loss,
            'D_loss': D_loss,
            'G_loss_GAN': G_loss_GAN,
            'G_loss_cyc': G_loss_cyc,
            'G_loss_rec': G_loss_rec,
            'D_acc': D_acc,
            'D_precision': D_precision,
            'D_recall': D_recall,
            'fake_evaluation': fake_evaluation,
            'recons_evaluation': recons_evaluation,
            'fake_target_img': fake_target_img,
            'fake_source_img': fake_source_img,
            'fake_source_recons_img': fake_source_recons_img
        }
        
        if self.disentangle:
            results.update({
                'R_loss': R_loss,
                'R_fake_target_v_source_acc': R_fake_target_v_source_acc,
                'R_fake_source_v_target_acc': R_fake_source_v_target_acc,
                'R_fake_source_v_source_acc': R_fake_source_v_source_acc,
                'R_fake_source_recon_v_source_acc': R_fake_source_recon_v_source_acc
            })
        
        return results
    
    def train_model(self, dataset):
        """
        Training loop for TensorFlow 2.x
        """
        # Setup tensorboard logging
        train_summary_writer = tf.summary.create_file_writer(self.log_dir)
        
        # Setup checkpointing
        # checkpoint = tf.train.Checkpoint(
        #     generator=self.G,
        #     discriminator=self.D,
        #     g_optimizer=self.G_opt,
        #     d_optimizer=self.D_opt
        # )
        # if self.disentangle:
        #     checkpoint.regularizer = self.R
        #     checkpoint.r_optimizer = self.R_opt

        # manager = tf.train.CheckpointManager(checkpoint, self.ckpt_dir, max_to_keep=5)

        # # Restore from checkpoint if available
        # if self.continue_train:
        #     checkpoint.restore(manager.latest_checkpoint)
        #     print(f"Restored from {manager.latest_checkpoint}")
        
        step_counter = 0
        
        for epoch in tqdm(range(self.EPOCHS)):
            print(f"Epoch {epoch + 1}/{self.EPOCHS}")
            
            for batch_idx, (images, labels) in enumerate(dataset):
                
                labels = tf.argmax(labels, axis=1)
                target_labels = np.random.randint(0, high=self.NUM_BINS, size=self.BATCH_SIZE)

                identity_ind = labels == target_labels

                labels = convert_ordinal_to_binary(labels, self.NUM_BINS)
                target_labels = convert_ordinal_to_binary(target_labels, self.NUM_BINS)

                if self.disentangle:
                    target_disentangle_ind = np.random.randint(0, high=self.k_dim, size=self.BATCH_SIZE)
                    target_disentangle_ind_one_hot = np.eye(self.k_dim)[target_disentangle_ind]
                    target_disentangle_ind_one_hot[identity_ind, :] = 0
                    y_regularizer = target_disentangle_ind
                    y_r = target_disentangle_ind_one_hot
                    
                # Train discriminator
                # if (batch_idx + 1) % self.discriminate_every_nth == 0:
                images = tf.cast(images, tf.float32)
                labels = tf.cast(labels, tf.float32)
                target_labels = tf.cast(target_labels, tf.float32)
                y_regularizer = tf.cast(y_regularizer, tf.float32)
                y_r = tf.cast(y_r, tf.float32)
                
                results = self.train_step(images, labels, target_labels, y_regularizer, y_r, train_phase=True)
                
                # Log discriminator results
                with train_summary_writer.as_default():
                    tf.summary.scalar('discriminator/loss_d', results['D_loss'], step=step_counter)
                    tf.summary.scalar('discriminator/loss_d_GAN', results['G_loss_GAN'], step=step_counter)
                    tf.summary.scalar('discriminator/acc_d', results['D_acc'], step=step_counter)
                    tf.summary.scalar('discriminator/precision_d', results['D_precision'], step=step_counter)
                    tf.summary.scalar('discriminator/recall_d', results['D_recall'], step=step_counter)
            
                # if (batch_idx + 1) % self.generate_every_nth == 0:
                #     # Log generator results
                #     with train_summary_writer.as_default():
                    tf.summary.scalar('generator/loss_g', results['G_loss'], step=step_counter)
                    tf.summary.scalar('generator/loss_g_GAN', results['G_loss_GAN'], step=step_counter)
                    tf.summary.scalar('generator/G_loss_cyc', results['G_loss_cyc'], step=step_counter)
                    tf.summary.scalar('generator/G_loss_rec', results['G_loss_rec'], step=step_counter)
                    tf.summary.scalar('generator/fake_evaluation', results['fake_evaluation'], step=step_counter)
                    tf.summary.scalar('generator/recons_evaluation', results['recons_evaluation'], step=step_counter)
                    
                    # Log images
                    tf.summary.image('real_img', images, step=step_counter, max_outputs=3)
                    tf.summary.image('fake_target_img', results['fake_target_img'], step=step_counter, max_outputs=3)
                    tf.summary.image('fake_source_img', results['fake_source_img'], step=step_counter, max_outputs=3)
                    tf.summary.image('fake_source_recons_img', results['fake_source_recons_img'], step=step_counter, max_outputs=3)
                    
                    tf.summary.scalar('disentangler/loss_r', results['R_loss'], step=step_counter)
                    tf.summary.scalar('disentangler/acc_r_fake_target_v_source', results['R_fake_target_v_source_acc'], step=step_counter)
                    tf.summary.scalar('disentangler/acc_r_fake_source_v_target', results['R_fake_source_v_target_acc'], step=step_counter)
                    tf.summary.scalar('disentangler/acc_r_fake_source_v_source', results['R_fake_source_v_source_acc'], step=step_counter)
                    tf.summary.scalar('disentangler/acc_r_fake_source_recon_v_source', results['R_fake_source_recon_v_source_acc'], step=step_counter)
            
                # Save samples
                sample_file = osp.join(self.sample_dir, '%06d.jpg' % step_counter)
                if step_counter % self.save_summary == 0:
                    save_images(results['fake_target_img'][:8], sample_file)
                
                # Save checkpoint
                if step_counter % self.save_ckpt == 0:
                    print('Checkpointing has to be rewritten!!!')
                    # save_path = manager.save()
                    # print(f"Saved checkpoint for step {step_counter}: {save_path}")
                
                step_counter += 1


"""def train():

    # ============= Load config =============
    config_path = 'D:\PycharmProjects\CF-Robustness-Benchmark\configs\dissect_derma.yaml' #args.config
    config = get_config(config_path)

    # ============= Experiment Folder=============
    assets_dir = os.path.join(config['log_dir'], config['name'])
    log_dir = os.path.join(assets_dir, 'log')
    ckpt_dir = os.path.join(assets_dir, 'checkpoints')
    sample_dir = os.path.join(assets_dir, 'sample')
    test_dir = os.path.join(assets_dir, 'test')
    # make directory if not exist
    try:
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(sample_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
    except Exception as e:
        print("Error in creating directories: ", e)

    # ============= Experiment Parameters =============
    ckpt_dir_cls = config.cls_experiment
    BATCH_SIZE = config.batch_size
    EPOCHS = config.epochs
    channels = config.num_channels
    input_size = config.data.img_size
    NUMS_CLASS_cls = config.data.num_classes
    NUMS_CLASS = config.num_bins
    target_class = config.target_class
    lambda_GAN = config.lambda_GAN
    lambda_cyc = config.lambda_cyc
    lambda_cls = config.lambda_cls
    save_summary = int(config.save_summary)
    save_ckpt = int(config.save_ckpt)
    ckpt_dir_continue = config.ckpt_dir_continue
    k_dim = config.k_dim
    lambda_r = config.lambda_r
    disentangle = k_dim > 1
    discriminate_every_nth = config.discriminate_every_nth
    generate_every_nth = config.generate_every_nth
    dataset = config.data.name


    model_weights = r'D:\PycharmProjects\CF-Robustness-Benchmark\notebooks\experiments\derma_classification\binary\checkpoints\resnet50_derma_acc65.keras'
    pretrained_classifier = tf.keras.models.load_model(model_weights) #celeba_classifier
    # my_data_loader = ImageLabelLoader(input_size=64)


    if ckpt_dir_continue == '':
        continue_train = False
    else:
        ckpt_dir_continue = os.path.join(ckpt_dir_continue, 'checkpoints')
        continue_train = True

    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    # ============= Data =============

    ds_builder = DatasetBuilder(config)
    ds_builder.setup()

    print('The size of the training set: ', ds_builder.train_dataset.data.info['n_samples']['train'])

    # ============= Save config =============
    with open(os.path.join(log_dir, 'setting.txt'), 'w') as fp:
        fp.write('config_file:' + str(config_path) + '\n')

    # ============= placeholder =============
    # x_source = tf.placeholder(tf.float32, [None, input_size, input_size, channels], name='x_source')
    # y_s = tf.placeholder(tf.int32, [None, NUMS_CLASS], name='y_s')
    # # TODO: ???
    # y_source = y_s[:, 0]
    # train_phase = tf.placeholder(tf.bool, name='train_phase')

    # y_t = tf.placeholder(tf.int32, [None, NUMS_CLASS], name='y_t')
    # y_target = y_t[:, 0]

    # if disentangle:
    #     y_regularizer = tf.placeholder(tf.int32, [None], name='y_regularizer')
    #     y_r = tf.placeholder(tf.float32, [None, k_dim], name='y_r')
    #     y_r_0 = tf.zeros_like(y_r, name='y_r_0')

    # ============= G & D =============
    G = Generator_Encoder_Decoder(name="generator")  # with conditional BN, SAGAN: SN here as well
    D = Discriminator_Ordinal(name="discriminator")  # with SN and projection
    R = Discriminator_Contrastive(name="disentangler")

        
    G_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0., beta_2=0.9)
    D_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0., beta_2=0.9)
    R_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0., beta_2=0.9)
    
    # ============= session =============
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    writer = tf.summary.FileWriter(log_dir, sess.graph)

    # ============= Checkpoints =============
    if continue_train:
        print(" [*] before training, Load checkpoint ")
        print(" [*] Reading checkpoint...")

        ckpt = tf.train.get_checkpoint_state(ckpt_dir_continue)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(ckpt_dir_continue, ckpt_name))
            print(ckpt_dir_continue, ckpt_name)
            print("Successful checkpoint upload")
        else:
            print("Failed checkpoint load")
    else:
        print(" [!] before training, no need to Load ")

    # ============= load pre-trained classifier checkpoint =============
    class_vars = [var for var in slim.get_variables_to_restore() if 'classifier' in var.name]
    name_to_var_map_local = {var.op.name: var for var in class_vars}
    temp_saver = tf.train.Saver(var_list=name_to_var_map_local)
    ckpt = tf.train.get_checkpoint_state(ckpt_dir_cls)
    ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
    temp_saver.restore(sess, os.path.join(ckpt_dir_cls, ckpt_name))
    print("Classifier checkpoint loaded.................")
    print(ckpt_dir_cls, ckpt_name)

    # ============= Training =============
    for e in range(EPOCHS):
        np.random.shuffle(data)
        for i in range(data.shape[0] // BATCH_SIZE):
            if args.debug:
                image_paths = np.array([str(ind) for ind in my_data_loader.tmp_list])
            else:
                image_paths = data[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
            img, labels = my_data_loader.load_images_and_labels(image_paths, image_dir=config['image_dir'], n_class=1,
                                                                file_names_dict=file_names_dict,
                                                                num_channel=channels, do_center_crop=True)

            labels = labels.ravel()
            target_labels = np.random.randint(0, high=NUMS_CLASS, size=BATCH_SIZE)

            identity_ind = labels == target_labels

            labels = convert_ordinal_to_binary(labels, NUMS_CLASS)
            target_labels = convert_ordinal_to_binary(target_labels, NUMS_CLASS)

            if disentangle:
                target_disentangle_ind = np.random.randint(0, high=k_dim, size=BATCH_SIZE)
                target_disentangle_ind_one_hot = np.eye(k_dim)[target_disentangle_ind]
                target_disentangle_ind_one_hot[identity_ind, :] = 0
                my_feed_dict = {y_t: target_labels, x_source: img, train_phase: True,
                                y_s: labels,
                                y_regularizer: target_disentangle_ind, y_r: target_disentangle_ind_one_hot}
            else:
                my_feed_dict = {y_t: target_labels, x_source: img, train_phase: True,
                                y_s: labels}

            if (i + 1) % discriminate_evert_nth == 0:

                _, d_loss, summary_str, counter = sess.run([D_opt, D_loss, d_sum, global_step],
                                                  feed_dict=my_feed_dict)
                writer.add_summary(summary_str, counter)

            if (i + 1) % generate_every_nth == 0:
                if disentangle:
                    _, g_loss, g_summary_str, r_loss, r_summary_str, counter = sess.run([G_opt, G_loss, g_sum, R_loss, r_sum, global_step],
                                                                               feed_dict=my_feed_dict)
                    # _, r_loss, r_summary_str = sess.run([R_opt, R_loss, r_sum], feed_dict=my_feed_dict)
                    writer.add_summary(r_summary_str, counter)
                else:
                    _, g_loss, g_summary_str, counter = sess.run([G_opt, G_loss, g_sum, global_step], feed_dict=my_feed_dict)
                writer.add_summary(g_summary_str, counter)

            def save_results(sess, step):
                num_seed_imgs = 8
                img, labels = my_data_loader.load_images_and_labels(image_paths[0:num_seed_imgs],
                                                                    image_dir=config['image_dir'], n_class=1,
                                                                    file_names_dict=file_names_dict,
                                                                    num_channel=channels,
                                                                    do_center_crop=True)
                labels = np.repeat(labels, NUMS_CLASS * k_dim, 0)
                labels = labels.ravel()
                labels = convert_ordinal_to_binary(labels, NUMS_CLASS)
                img_repeat = np.repeat(img, NUMS_CLASS * k_dim, 0)

                target_labels = np.asarray([np.asarray(range(NUMS_CLASS)) for j in range(num_seed_imgs * k_dim)])
                target_labels = target_labels.ravel()
                identity_ind = labels == target_labels
                target_labels = convert_ordinal_to_binary(target_labels, NUMS_CLASS)

                if disentangle:
                    target_disentangle_ind = np.asarray(
                        [np.repeat(np.asarray(range(k_dim)), NUMS_CLASS) for j in range(num_seed_imgs)])
                    target_disentangle_ind = target_disentangle_ind.ravel()
                    target_disentangle_ind_one_hot = np.eye(k_dim)[target_disentangle_ind]
                    target_disentangle_ind_one_hot[identity_ind, :] = 0
                    my_feed_dict = {y_t: target_labels, x_source: img_repeat, train_phase: False,
                                    y_s: labels,
                                    y_regularizer: target_disentangle_ind, y_r: target_disentangle_ind_one_hot}
                else:
                    my_feed_dict = {y_t: target_labels, x_source: img_repeat, train_phase: False,
                                    y_s: labels}

                FAKE_IMG, fake_logits_ = sess.run([fake_target_img, fake_target_logits],
                                                  feed_dict=my_feed_dict)

                output_fake_img = np.reshape(FAKE_IMG, [-1, k_dim, NUMS_CLASS, input_size, input_size, channels])

                # save samples
                sample_file = os.path.join(sample_dir, '%06d.jpg' % step)
                save_images(output_fake_img, sample_file, num_samples=num_seed_imgs,
                            nums_class=NUMS_CLASS, k_dim=k_dim, image_size=input_size, num_channel=channels)
                np.save(sample_file.split('.jpg')[0] + '_y.npy', labels)

            _approx_num_seen_batches = int(counter/3)
            if _approx_num_seen_batches % save_summary == 0:
                save_results(sess, _approx_num_seen_batches)

            if _approx_num_seen_batches % save_ckpt == 0:
                saver.save(sess, ckpt_dir + "/model%2d.ckpt" % _approx_num_seen_batches, global_step=global_step)
"""

if __name__ == "__main__":
    # train()

    config_path = 'D:\PycharmProjects\CF-Robustness-Benchmark\configs\dissect_derma.yaml'
    config = get_config(config_path)

    # Initialize the dataset builder
    ds_builder = DatasetBuilder(config)
    ds_builder.setup()

    
    train_images = ds_builder.train_dataset.data.imgs / 255.0

    train_labels = to_categorical(ds_builder.train_dataset.data.labels)

    val_images = ds_builder.val_dataset.data.imgs / 255.0
    val_labels = to_categorical(ds_builder.val_dataset.data.labels)

    test_images = ds_builder.test_dataset.data.imgs / 255.0
    test_labels = to_categorical(ds_builder.test_dataset.data.labels)

    train_images_comb = np.concatenate([train_images, val_images], axis=0)
    train_labels_comb = np.concatenate([train_labels, val_labels], axis=0)

    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(train_labels.shape[0]).batch(config.batch_size)
    val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_labels)).shuffle(val_labels.shape[0]).batch(config.batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(config.batch_size)

    # Initialize the trainer
    trainer = DiscovererTrainer(config)
    trainer.initialize_networks()
    trainer.initialize_optimizers()


    trainer.train_model(train_ds)
