import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
from sklearn.model_selection import StratifiedKFold, train_test_split
from my_fuctions.data_load import make_test_data
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from my_fuctions.value import *
from my_fuctions.models import *
from sklearn import metrics


path = "dog/eye/ultrasound"

x, y = make_test_data(path, 'cataract')

image_shape = (256, 256, 3)
classes = 2

print(x.shape)
print(y.shape)



x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.2, random_state=1337, stratify=y)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.2, random_state=1337, stratify=y_temp)

y_train = to_categorical(y_train, num_classes=classes)
y_val = to_categorical(y_val, num_classes=classes)
y_test = to_categorical(y_test, num_classes=classes)



def train_and_evaluate_model(model, model_name, x_train, y_train, x_val, y_val, x_test, y_test, last_layer):
  precision = tf.keras.metrics.Precision()
  recall = tf.keras.metrics.Recall()
  auc = tf.keras.metrics.AUC()
  f1_score = tfa.metrics.F1Score(num_classes=classes, threshold=0.5)


  model.summary()
  model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy", precision, recall, auc, f1_score])

  #model.fit(train_datagen.flow(x_train, y_train, batch_size=32), epochs = epochs, validation_data=(x_val, y_val))

  #unfreeze_model(model, 10)
    
  history = model.fit(x_train, y_train, batch_size=32, epochs=epochs, validation_data=(x_val, y_val)
      #callbacks=callbacks
  )

  result = model.evaluate(x_test, y_test, verbose=2)
  test_loss, test_acc = result[0], result[1]
  
  roc(x_test, y_test, model, model_name)
  # pr(x_test, y_test, model, model_name)

  model.layers[-1].activation = None
  
  heatmap = make_gradcam_heatmap(x_test, model, last_layer)
  save_htm(x_test[-1], heatmap[-1], model_name)

  saliency = make_sal(x_test, model)
  save_sal(x_test[-1], saliency[-1], model_name)
  
  img_array = x_test[-1].transpose(2, 0, 1) * 255.0
  lime(img_array, model, model_name)
  
  h_odd, s_odd = odd_ratio(heatmap, saliency, y_test)
  
  h_odd = round(h_odd, 3)
  s_odd = round(s_odd, 3)

  print('htm : ', h_odd)
  print('sal : ', s_odd)

  report(x_test, y_test, model, model_name)

  test_acc = round(test_acc, 3)
  test_loss = round(test_loss, 3)

  print('accuracy : ', test_acc)
  print('loss : ', test_loss)
  
  test_result = model.predict(x_test)
  y_pred = np.argmax(test_result, axis=1)
  y_true = np.argmax(y_test, axis=1)

  recall = metrics.recall_score(y_true, y_pred, average='micro')
  precision = metrics.precision_score(y_true, y_pred, average='micro')
  f1_score = metrics.f1_score(y_true, y_pred, average='micro')
  auc = metrics.roc_auc_score(y_test, test_result, multi_class='ovr')

  recall = round(recall, 3)
  precision = round(precision, 3)
  f1_score = round(f1_score, 3)
  auc = round(auc, 3)

  save_csv(model_name, test_acc, test_loss, h_odd, s_odd)
  save_met(model_name, recall, precision, f1_score, auc)
  
  print("recall :", recall)
  print("precision :", precision)
  print("f1 score :", f1_score)
  print("auc :", auc)
  #return test_acc, test_loss, h_odd, s_odd, model

layer_list = ['last_layer', 'block5_pool', 'block5_pool', 'post_relu', 'relu', 'activation_259', 'conv_7b_ac', 'top_activation', 'top_activation', 'layer_normalization']
layer_list = ['block5_pool', 'block5_pool', 'post_relu', 'relu', 'activation_259', 'conv_7b_ac', 'top_activation', 'top_activation', 'layer_normalization']


epochs = 10

model_d = {
    #'CNN': make_model(image_shape, num_classes=classes),
    'VGG16' : make_model4(image_shape, num_classes=classes),
    'VGG16_1' : make_model4_1(image_shape, num_classes=classes),
    'ResNet50V2' : make_model5(image_shape, num_classes=classes),
    'DenseNet121' : make_model6(image_shape, num_classes=classes),
    'NASNetLarge' : make_model7(image_shape, num_classes=classes),
    'InceptionResNetV2' : make_model8(image_shape, num_classes=classes),
    'EfficientNetV2B0' : make_model9(image_shape, num_classes=classes),
    'EfficientNetB0' : make_model10(image_shape, num_classes=classes),
    'ConvNeXtTiny' : make_model11(image_shape, num_classes=classes)
}


for (model_name, model), layer in zip(model_d.items(), layer_list):
  print(model_name)
  train_and_evaluate_model(model, model_name, x_train, y_train, x_val, y_val, x_test, y_test, layer)

