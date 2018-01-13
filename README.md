# vehicle-detection-and-tracking

From the lecture notes, here are some useful functions:

`cv2.rectangle(image_to_draw_on, (x1, y1), (x2, y2), color, thick)`  
usage example:
```python
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    draw_img = np.copy(img)
    for box in bboxes:
        (x1, y1) = box[0]
        (x2, y2) = box[1]
        cv2.rectangle(draw_img, (x1, y1), (x2, y2), color, thick)
    return draw_img # Change this line to return image copy with boxes
```
