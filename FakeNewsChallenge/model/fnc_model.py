import tensorflow as tf


class FNCModel:
    """
    A Tensorflow v2 implementation of the model in https://github.com/uclnlp/fakenewschallenge
    """
    model = None
    hyperparameters = None

    def __init__(self, hyperparameters):
        self.hyperparameters = hyperparameters
        # Define multi-layer perceptron
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(hyperparameters["hidden_size"], activation="relu",
                                  kernel_regularizer=tf.keras.regularizers.l2(hyperparameters["l2_alpha"])),
            tf.keras.layers.Dropout(hyperparameters["dropout"]),
            tf.keras.layers.Dense(hyperparameters["target_size"],
                                  kernel_regularizer=tf.keras.regularizers.l2(hyperparameters["l2_alpha"])),
            tf.keras.layers.Dropout(hyperparameters["dropout"]),
            tf.keras.layers.Reshape((hyperparameters["batch_size"], hyperparameters["target_size"])),
            tf.keras.layers.Softmax(),
        ])

        # define optimizer and loss
        optimizer = tf.keras.optimizers.Adam(learning_rate=hyperparameters["learning_rate"],
                                             global_clipnorm=hyperparameters["clip_ratio"])
        loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                  reduction=tf.keras.losses.Reduction.SUM)
        self.model.compile(
            optimizer=optimizer,
            loss=loss_func,
            metrics=['accuracy']
        )

        print("Model initialized.")

    def train_then_eval(self, train_dataset, val_dataset, test_dataset, test_run_mode=False):
        print("Model training started.")
        res = self.model.fit(
            train_dataset if test_run_mode else train_dataset[:4],
            validation_data=val_dataset if not test_run_mode else train_dataset[:2],
            epochs=self.hyperparameters["epochs"],
        )
        print("Model training finished.")
        print(f"Model Summary: \n{self.model.summary()}")
        test_loss, test_acc = self.model.evaluate(test_dataset)

        print(f"Test loss: {test_loss} | Test Acc: {test_acc}")
        print(f"Run has completed with hyperparameters: \n{self.hyperparameters}")
