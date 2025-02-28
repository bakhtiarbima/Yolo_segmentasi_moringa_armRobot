import cv2
import numpy as np
import serial
import time
from ultralytics import YOLO

# Inisialisasi komunikasi serial ke Arduino
ser = serial.Serial('COM10', 115200, timeout=1)  # Sesuaikan dengan COM port Arduino
time.sleep(2)  # Tunggu agar koneksi serial stabil

# Muat model YOLO
model = YOLO("best.pt")

# Buka kamera
cap = cv2.VideoCapture(0)

frame_count = 0
segmentation_interval = 10  # Perbarui segmentasi setiap beberapa frame
real_box_size = 25  # Ukuran kotak di dunia nyata (cm)

# Status komunikasi dengan Arduino
waiting_for_arduino = False

# Label objek yang boleh diproses
allowed_labels = ["moringa", "dried-leaves"]

def detect_red_rectangle(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 | mask2
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        epsilon = 0.05 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 4:
            return approx
    return None

def clean_mask(mask, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=1)
    cleaned = cv2.erode(dilated, kernel, iterations=1)
    return cleaned

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame dari kamera.")
        break

    if waiting_for_arduino:
        while True:
            arduino_response = ser.readline().decode().strip()
            if arduino_response:
                print(f"Respon dari Arduino: {arduino_response}")
                if "arduino selesai" in arduino_response.lower():
                    print("Arduino selesai, melanjutkan segmentasi.")
                    waiting_for_arduino = False
                    break
            time.sleep(0.1)

    original_frame = frame.copy()
    red_rect = detect_red_rectangle(frame)

    if red_rect is not None:
        rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(red_rect)
        pts1 = np.float32(red_rect.reshape(4, 2))
        pts2 = np.float32([[0, 0], [rect_w, 0], [rect_w, rect_h], [0, rect_h]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        warped_frame = cv2.warpPerspective(frame, matrix, (rect_w, rect_h))

        cv2.polylines(original_frame, [red_rect], isClosed=True, color=(0, 255, 0), thickness=2)

        # **Hitung ulang skala berdasarkan ukuran hasil warp**
        px_to_cm_x = real_box_size / warped_frame.shape[1]
        px_to_cm_y = real_box_size / warped_frame.shape[0]

        if frame_count % segmentation_interval == 0:
            results = model.predict(source=warped_frame, conf=0.5, device='cpu', save=False, show=False)
            segmentation_done = False

            if results and results[0].masks is not None:
                for result in results:
                    for box, mask, cls in zip(result.boxes.xyxy, result.masks.data.numpy(), result.boxes.cls):
                        class_name = result.names[int(cls)]  # Ambil nama kelas
                        
                        if class_name not in allowed_labels:
                            continue  # Skip jika bukan moringa atau dried leaves
                        
                        # **Lanjutkan proses segmentasi**
                        mask_resized = cv2.resize(mask, (warped_frame.shape[1], warped_frame.shape[0]))
                        mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
                        cleaned = clean_mask(mask_binary)

                        # Overlay hasil segmentasi
                        mask_colored = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
                        warped_frame = cv2.addWeighted(warped_frame, 0.7, mask_colored, 0.3, 0)

                        # **Hitung titik tengah**
                        moments = cv2.moments(cleaned)
                        if moments["m00"] != 0:
                            cx = int(moments["m10"] / moments["m00"])
                            cy = int(moments["m01"] / moments["m00"])
                            cx_cm = cx * px_to_cm_x
                            cy_cm = cy * px_to_cm_y

                            # **Gambar titik tengah**
                            cv2.circle(warped_frame, (cx, cy), 10, (0, 0, 255), -1)
                            cv2.putText(warped_frame, f"({cx_cm:.2f}cm, {cy_cm:.2f}cm)",
                                        (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                            print(f"Deteksi {class_name}: ({cx_cm:.2f}cm, {cy_cm:.2f}cm)")

                            # **Konversi ke integer sebelum dikirim ke Arduino**
                            cx_cm_scaled = int(cx_cm + 7)
                            cy_cm_scaled = int(cy_cm - 5)

                            # **Tampilkan hasil segmentasi sebelum mengirim data**
                            cv2.imshow("Processed Frame", cv2.resize(warped_frame, (640, 544)))
                            cv2.waitKey(500)  # Pause sebentar agar tampilan bisa terlihat

                            # **Kirim koordinat ke Arduino**
                            ser.write(f"{cx_cm_scaled},{cy_cm_scaled}\n".encode())

                            # **Tunggu respon dari Arduino**
                            while True:
                                arduino_response = ser.readline().decode().strip()
                                if arduino_response:
                                    print(f"Respon dari Arduino: {arduino_response}")
                                if "arduino selesai" in arduino_response.lower():
                                    break

                            segmentation_done = True

            if segmentation_done:
                processed_frame = warped_frame.copy()
            else:
                processed_frame = original_frame.copy()

    else:
        cv2.putText(original_frame, "ROI not detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        print("ROI not detected")
        processed_frame = original_frame.copy()

    cv2.imshow("Webcam", original_frame)
    cv2.imshow("Processed Frame", cv2.resize(processed_frame, (640, 544)))

    frame_count += 1

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
ser.close()
