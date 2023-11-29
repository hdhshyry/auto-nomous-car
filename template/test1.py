import cv2
import numpy as np

def detect_sign(image_path, template_path):
    # خواندن تصویر و قالب
    image = cv2.imread(image_path)
    template = cv2.imread(template_path)

    # ابعاد قالب
    template_height, template_width, _ = template.shape
    image= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # تطابق الگو
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    
    # دریافت موقعیتهای تطابق بالا
    threshold = 0.8
    locations = np.where(result >= threshold)
    if locations is not None:
        for loc in locations:
            # دریافت موقعیت بالا
            top_left = loc[0]
            bottom_right = (top_left[0] + template_width, top_left[1] + template_height)

            # نمایش مستطیل بر روی تصویر اصلی
            cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

        # نمایش تصویر با علائم تشخیص داده شده
        cv2.imshow("Detected Signs", image)
        cv2.waitKey(0)
    else:
        print("No signs found.")

# مسیر تصویر و قالب
image_path = "1.png"
template_path = "dead.png"

# تشخیص علائم
detect_sign(image_path, template_path)