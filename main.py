import argparse
from network import VGG16Classifier
import numpy as np
from processing import ImageDataLoader
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


if __name__ == "__main__":
    parser = argparse.ArgumentParser('main.py to train the model',usage='python main.py data --num_classes 2 --image_size 150 --epochs 1')
    parser.add_argument('data',type=str,help='data')
    parser.add_argument('--num_classes',type=int,help='Number of classes')
    parser.add_argument('--image_size',type=int,help='Size of the images')
    parser.add_argument('--epochs',type=int,help='number of epochs')

    args = parser.parse_args()
    data = args.data
    n_classes = args.num_classes
    size = args.image_size
    epochs = args.epochs

    loader = ImageDataLoader(image_path=data, image_size=size)
    loader.preprocess_data()
    X_train, X_test, Y_train, Y_test = loader.shuffle_and_split()

    classifier = VGG16Classifier(input_shape=(size, size, 3), num_classes=n_classes)
    history = classifier.train(X_train, Y_train, X_test, Y_test, epochs=epochs, batch_size=10)


    loss, accuracy = classifier.evaluate(X_train, Y_train)
    print("\n-------------------Train Loss:----------------- ", loss)
    print("\n------------------Train Accuracy:-------------- ", accuracy)
    loss, accuracy = classifier.evaluate(X_test, Y_test)
    print("\n-------------------Test Loss:------------------ ", loss)
    print("\n------------------Test Accuracy:--------------- ", accuracy)


    predictions = classifier.predict(X_test)
    y_pred=np.argmax(predictions, axis=1)
    y_test=np.argmax(Y_test, axis=1)
    cm = confusion_matrix(y_test, y_pred)
    print("\n-----------------Confusion Matrix---------------")
    print(cm)
    print("\n---------------Classification Report-------------")
    print(classification_report(y_test, y_pred))

