import shutil

import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras import Model
from tensorflow.keras.metrics import Mean, BinaryAccuracy
import pickle as pkl

from src.Models.discriminator import Discriminator
from src.Models.generator import Generator


class GAN(Model):
    def __init__(
            self,
            hidden_dim,
            noise_shape,
            num_words,
            dim_head=64,
            num_heads=1,
            dropout_rate=0.2,
            num_layers=3
    ):
        super(GAN, self).__init__()

        self.hidden_dim = hidden_dim
        self.noise_shape = noise_shape
        self.num_words = num_words
        self.dim_head = dim_head
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers

        self.generator_loss_tracker = Mean(name="generator_loss")
        self.discriminator_loss_tracker = Mean(name="discriminator_loss")
        self.discriminiator_acc_tracker = BinaryAccuracy(name="discriminator_acc")

        self.build()

    @property
    def metrics(self):
        return [
            self.generator_loss_tracker,
            self.discriminator_loss_tracker,
            self.discriminiator_acc_tracker
        ]

    def build(self, **kwargs):
        self.cross_entropy = BinaryCrossentropy(from_logits=True)

        self.discriminator = Discriminator(
            self.hidden_dim, 2,
            self.dim_head, self.num_heads,
            self.dropout_rate, self.num_layers
        )

        self.generator = Generator(
            self.hidden_dim, self.noise_shape,
            self.num_words, self.hidden_dim,
            self.dim_head, self.num_heads,
            self.dropout_rate, self.num_layers
        )

        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-5)

    @tf.function
    def train_step(self, text_batch):
        batch_size = text_batch.shape[0]
        print(text_batch.shape)
        text_one_hot = tf.one_hot(text_batch, self.num_words)
        rand_noise = tf.random.uniform(text_one_hot.shape, maxval=0.2)
        text_one_hot += rand_noise
        text_one_hot = tf.nn.softmax(text_one_hot, axis=-1)

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_text = self.generator.generate(batch_size, training=True)

            real_classification = self.discriminator(text_one_hot, training=True)
            fake_classification = self.discriminator(generated_text, training=True)

            generator_loss = self.generator_loss(
                fake_classification
            )
            discriminator_loss = self.discriminator_loss(
                real_classification, fake_classification
            )

        generator_gradients = gen_tape.gradient(
            generator_loss, self.generator.trainable_variables
        )
        discriminator_gradients = disc_tape.gradient(
            discriminator_loss, self.discriminator.trainable_variables
        )

        self.generator_optimizer.apply_gradients(
            zip(generator_gradients, self.generator.trainable_variables)
        )
        self.discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, self.discriminator.trainable_variables)
        )

        self.generator_loss_tracker.update_state(generator_loss)
        self.discriminator_loss_tracker.update_state(discriminator_loss)

        return {
            "generator_loss": self.generator_loss_tracker.result(),
            "discriminator_loss": self.discriminator_loss_tracker.result(),
            "discriminator_acc": self.discriminiator_acc_tracker.result()
        }

    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        self.discriminiator_acc_tracker.update_state(tf.ones_like(real_output), real_output)

        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        self.discriminiator_acc_tracker.update_state(tf.zeros_like(fake_output), fake_output)

        total_loss = real_loss + fake_loss
        return total_loss

    def generate(self, batch_size=1):
        return self.generator.generate(batch_size)

    def get_config(self):
        config = super(Generator, self).get_config()
        config.update({
            "hidden_dim": self.hidden_dim, "noise_shape": self.noise_shape,
            "num_words": self.num_words, "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate, "num_layers": self.num_layers,
            "num_classes": self.num_classes
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def save_gan(self, filename="model_save.h5"):
        g_weights = self.generator.get_weights()
        d_weights = self.discriminator.get_weights()

        with open(filename, "wb") as f:
            pkl.dump([g_weights, d_weights], f)

        if filename[-1:] == '9':
            print("incerc sa salvez")
            #  with open('/content/gdrive/My Drive/gan_poezii/filename', 'w') as f:
            #   f.write(weights)
            shutil.copy(filename, '/content/drive/My Drive/gan_poezii/')

    def load_gan(self, filename="model_save.h5"):
        with open(filename, "rb") as f:
            weights = pkl.load(f)

        mock_batch = tf.zeros((1, self.num_words), dtype=tf.int32)
        self.fit(mock_batch, batch_size=1)

        self.generator.set_weights(weights[0])
        self.discriminator.set_weights(weights[1])
