import tensorflow as tf
import numpy as np

XOR_TRAINING = "train.csv"

XOR_TEST = "test.csv"


def main():

    #Load Dataset:
    training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=XOR_TRAINING,
        target_dtype=np.int,
        features_dtype=np.float32)

    test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=XOR_TEST,
        target_dtype=np.int,
        features_dtype=np.float32)

    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=2)]

    classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                              hidden_units=[5, 200, 10],
                                              n_classes=2,
                                              model_dir="/tmp/xor_model")


    # Define the training inputs
    def get_train_inputs():
        x = tf.constant(training_set.data)
        y = tf.constant(training_set.target)

        return x, y

    # Fit model.
    classifier.fit(input_fn=get_train_inputs, steps=1000)

    # Define the test inputs
    def get_test_inputs():
        x = tf.constant(test_set.data)
        y = tf.constant(test_set.target)

        return x, y

    accuracy_score = classifier.evaluate(input_fn=get_test_inputs,
                                       steps=1)["accuracy"]

    print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

    # Classify two new flower samples.
    def new_samples():
        return np.array(
        [[0,0],
            [1,1]], dtype=np.float32)

    predictions = list(classifier.predict(input_fn=new_samples))
    print(
        "New Samples, Class Predictions:    {}\n"
        .format(predictions))

main()