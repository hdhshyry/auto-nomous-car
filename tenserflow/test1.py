import cv2
import tensorflow as tf
from object_detection.utils import label_map_util

# Load the pre-trained model
model_path = "path/to/your/model"
model = tf.saved_model.load(model_path)

# Load label map
label_map_path = "path/to/your/label_map.pbtxt"
category_index = label_map_util.create_category_index_from_labelmap(label_map_path, use_display_name=True)

# Load and preprocess the image
image = cv2.imread("path/to/your/image.jpg")
image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
input_tensor = tf.convert_to_tensor([image_np])

# Perform inference
detections = model(input_tensor)

# Visualize the results
for score, box, class_id in zip(detections['detection_scores'][0], detections['detection_boxes'][0], detections['detection_classes'][0]):
    if score > 0.5:  # Adjust confidence threshold as needed
        class_name = category_index[int(class_id)]['name']
        ymin, xmin, ymax, xmax = box.numpy()
        xmin, xmax, ymin, ymax = int(xmin * image.shape[1]), int(xmax * image.shape[1]), int(ymin * image.shape[0]), int(ymax * image.shape[0])
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(image, f"{class_name}: {score}", (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the result
cv2.imshow("Result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()