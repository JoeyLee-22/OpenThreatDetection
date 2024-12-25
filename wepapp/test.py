import numpy as np
import tensorflow as tf
import cv2

from PIL import Image
from wepcore.inference_images_weapon import inference_images_weapon

img = Image.open('gun.jpg')
image_mask1 = np.array(img)

# (image2,start_time,end_time,scores,classes) = inference_images_weapon(img, "test", 10)

path = 'wepapp/weaponresource/checkpoints_weapon/WeaponOct24_608_8K'
detect_weapon = tf.saved_model.load(path)

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
    iou_threshold=iou_weapon,
    score_threshold=score_weapon
)
# format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
original_h, original_w, _ = image_mask1.shape
bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)

# hold all detection data in one variable
pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]

# rmm
class_names = utils.read_class_names("./wepdata/classes/weapons.names")
# print(class_names)

# by default allow all classes in .names file
allowed_classes = list(class_names.values())
#end_time3 = time.time()


# if crop flag is enabled, crop each detection and save it as new image
if crop:
    crop_rate = cfg.crop_rate # capture images every so many frames (ex. crop photos every 150 frames)
    crop_path = os.path.join(os.getcwd(), 'detections', 'crop', video_name)
    try:
        os.mkdir(crop_path)
    except FileExistsError:
        pass
    if frame_num % crop_rate == 0:
        final_path = os.path.join(crop_path, 'frame_' + str(frame_num))
        try:
            os.mkdir(final_path)
        except FileExistsError:
            pass
        crop_objects(cv2.cvtColor(image_mask1, cv2.COLOR_BGR2RGB), pred_bbox, final_path, allowed_classes)
    else:
        pass


# if count flag is enabled, perform counting of objects
if count:
    # count objects found
    counted_classes = count_objects(pred_bbox, by_class = False, allowed_classes=allowed_classes)
    # loop through dict and print
    for key, value in counted_classes.items():
        log.debug("Number of {}s: {}".format(key, value))
    image2 = utils.draw_bbox(image_mask1, pred_bbox, info, counted_classes, allowed_classes=allowed_classes, read_plate = plate)
else:
    image2 = utils.draw_bbox(image_mask1, pred_bbox, info, allowed_classes=allowed_classes, read_plate = plate)

image = Image.fromarray(image2)
image.save("gun2.jpg")