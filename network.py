import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model

class VGG16Classifier:
    def __init__(self, input_shape=(150, 150, 3), num_classes=5):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()
    
    def build_model(self):
        vgg = VGG16(input_shape=self.input_shape, weights='imagenet', include_top=False)

        for layer in vgg.layers:
            layer.trainable = False

        x = Flatten()(vgg.output)
        prediction = Dense(self.num_classes, activation='softmax')(x)

        model = Model(inputs=vgg.input, outputs=prediction)
        model.compile(optimizer='adam',
                      loss=tf.losses.CategoricalCrossentropy(),
                      metrics=[keras.metrics.AUC(name='auc')])
        return model
    
    def train(self, X_train, Y_train, X_test, Y_test, epochs=50, batch_size=10):
        print("-------------------Started training------------------\n")

        callback = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                 patience=8,
                                                 restore_best_weights=True)
        history = self.model.fit(X_train, Y_train,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 validation_data=(X_test, Y_test),
                                 callbacks=[callback])
        return history
    def predict(self, X):
        return self.model.predict(X)
    def evaluate(self, X, Y):
        return self.model.evaluate(X, Y)
