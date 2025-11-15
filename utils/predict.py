import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
from PIL import Image
import segmentation_models as sm

# Loss functions required for loading your model
weights = [0.166]*6
dice_loss = sm.losses.DiceLoss(class_weights=weights)
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + focal_loss

def jaccard_coef(y_true, y_pred):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (tf.keras.backend.sum(y_true_f) +
                                   tf.keras.backend.sum(y_pred_f) -
                                   intersection + 1.0)

def load_segmentation_model():
    model = load_model(
        "model/Model_satellite_segmentation.h5",
        custom_objects={
            "dice_loss_plus_1focal_loss": total_loss,
            "jaccard_coef": jaccard_coef
        }
    )
    return model

def predict_mask(model, image_np):
    img_resized = cv2.resize(image_np, (256, 256))
    img_norm = img_resized / 255.0
    img_norm = np.expand_dims(img_norm, axis=0)

    pred = model.predict(img_norm)[0]
    mask = np.argmax(pred, axis=-1)

    mask = cv2.resize(mask.astype("uint8"),
                      (image_np.shape[1], image_np.shape[0]))

    return mask
