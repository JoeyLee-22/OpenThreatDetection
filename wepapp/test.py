import numpy as np
import tensorflow as tf
import cv2
import wepcore.utils as utils

cap = cv2.VideoCapture(0)

path = 'weaponresource/checkpoints_weapon/WeaponOct24_608_8K'
detect_weapon = tf.saved_model.load(path)

while True:
    success, image_mask1 = cap.read()
    
    image_data = cv2.resize(image_mask1, (608, 608))
    image_data = image_data / 255.
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    infer_weapon = detect_weapon.signatures['serving_default']

    batch_data = tf.constant(image_data)
    pred_bbox = infer_weapon(batch_data)

    for key, value in pred_bbox.items():
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]

    # run non max suppression on detections
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=0.5,
        score_threshold=0.3
    )

    original_h, original_w, _ = image_mask1.shape
    bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)

    pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]

    class_names = utils.read_class_names("./wepdata/classes/weapons.names")

    allowed_classes = list(class_names.values())

    image2 = utils.draw_bbox(image_mask1, pred_bbox, info = False, allowed_classes=allowed_classes)
    
    cv2.imshow('Webcam', image2)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()