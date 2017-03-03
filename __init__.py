import tensorflow as tf

from os.path import expanduser
from datetime import datetime

MODELS_PATH = expanduser("~") + '/' + 'tf_models'


def save_state(session, name=None):
    name = name or datetime.now().isoformat()
    saver = tf.train.Saver()
    path = '{}/{}.ckpt'.format(MODELS_PATH, name)
    saver.save(session, path)
    return path, name


def load_state(session, name):
    saver = tf.train.Saver()
    path = '{}/{}.ckpt'.format(MODELS_PATH, name)
    saver.restore(session, path)
