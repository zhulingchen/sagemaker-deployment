import argparse
import json
import os
import pickle
import sys
import pandas as pd
import tensorflow as tf

from model import DCNN



def _get_train_ds(batch_size, training_dir):
    print("Get train dataset.")

    train_data = pd.read_csv(os.path.join(training_dir, 'train.csv'), header=None, names=None)

    # Turn the input pandas dataframe into numpy arrays
    train_y = train_data[[0]].values.squeeze().astype('float')
    train_X = train_data.drop([0], axis=1).values

    # build TF dataset
    train_ds = tf.data.Dataset.from_tensor_slices((train_X, train_y))
    return train_ds.shuffle(100000, reshuffle_each_iteration=True).batch(batch_size)


def train(model, train_dataset, epochs, optimizer, loss_fn):
    for epoch in range(1, epochs + 1):
        total_loss = tf.keras.metrics.Mean(name='train_loss')
        for batch_X, batch_y in train_dataset:
            with tf.GradientTape() as tape:
                forward_y = model(batch_X, training=True)
                loss = loss_fn(forward_y, batch_y)
            tvars = model.trainable_variables
            gradients = tape.gradient(loss, tvars)
            optimizer.apply_gradients(zip(gradients, tvars))            
            total_loss(loss)
        print("Epoch: {}, BCELoss: {}".format(epoch, total_loss.result()))



if __name__ == '__main__':
    # All of the model parameters and training parameters are sent as arguments when the script is executed.
    # Here we set up an argument parser to easily access the parameters.

    # print some environment variables
    print("TensorFlow Version:", tf.__version__)
    print("os.environ['SM_HOSTS'] =", os.environ['SM_HOSTS'])
    print("os.environ['SM_CURRENT_HOST'] =", os.environ['SM_CURRENT_HOST'])
    print("os.environ['SM_MODEL_DIR'] =", os.environ['SM_MODEL_DIR'])
    print("os.environ['SM_CHANNEL_TRAINING'] =", os.environ['SM_CHANNEL_TRAINING'])
    print("os.environ['SM_NUM_GPUS'] =", os.environ['SM_NUM_GPUS'])
    
    parser = argparse.ArgumentParser()

    # Training Parameters
    parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 512)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    # Model Parameters
    parser.add_argument('--embedding_dim', type=int, default=128, metavar='N',
                        help='size of the word embeddings (default: 128)')
    parser.add_argument('--filter_number', type=int, default=64, metavar='N',
                        help='number of 1D CNN filters (default: 64)')
    parser.add_argument('--dense_units', type=int, default=512, metavar='N',
                        help='number of units in the dense layer (default: 512)')
    parser.add_argument('--vocab_size', type=int, default=5000, metavar='N',
                        help='size of the vocabulary (default: 5000)')

    # SageMaker Parameters
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model_dir', type=str)
    # TensorFlow 2 uses Script Mode so the trained model will be saved to the path defined in args.sm_model_dir
    # https://stackoverflow.com/questions/60190365/sagemaker-with-tensorflow-2-not-saving-model
    parser.add_argument('--sm-model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    args = parser.parse_args()
    print("args =", args)

    print("Is CUDA available?", tf.test.is_gpu_available())
    
    tf.random.set_seed(args.seed)

    # Load the training data.
    train_ds = _get_train_ds(args.batch_size, args.data_dir)

    # Build the model.    
    model = DCNN(vocab_size=args.vocab_size,
                 emb_dim=args.embedding_dim,
                 nb_filters=args.filter_number,
                 FFN_units=args.dense_units)

    # Train the model.
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    train(model, train_ds, args.epochs, optimizer, loss_fn)
    model.summary()
    print("Model is trained.")

    # Load the word_dict from the data directory
    with open(os.path.join(args.data_dir, "word_dict.pkl"), "rb") as f:
        word_dict = pickle.load(f)
    
	# Save the word_dict to the model directory
    word_dict_path = os.path.join(args.sm_model_dir, 'word_dict.pkl')
    with open(word_dict_path, 'wb') as f:
        pickle.dump(word_dict, f)

	# Save the model to Tensorflow SavedModel
    # need a version number folder in the path
    # https://stackoverflow.com/questions/59882941/valueerror-no-savedmodel-bundles-found-when-trying-to-deploy-a-tf2-0-model-to
    model.predict(next(iter(train_ds))[0])
    model.save(os.path.join(args.sm_model_dir, 'model', '1'), save_format='tf')