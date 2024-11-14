from flask import Flask, request, jsonify, render_template, send_from_directory
import cv2
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)

# 결과 이미지를 저장할 임시 폴더 경로
TEMP_FOLDER = "temp_results"
os.makedirs(TEMP_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/count_colonies', methods=['POST'])
def count_colonies():
    colony_counts = []
    images = []

    for image_file in request.files.getlist('images'):
        # 이미지 읽기
        image = Image.open(io.BytesIO(image_file.read())).convert('RGB')
        image_np = np.array(image)

        # 이미지를 HSV 색 공간으로 변환
        hsv_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)

        # 형광 콜로니에 대한 HSV 임계값 설정
        fluorescent_mask = cv2.inRange(hsv_image, (30, 100, 100), (90, 255, 255))
        non_fluorescent_mask = cv2.bitwise_not(fluorescent_mask)

        # 형광 및 비형광 콜로니 검출
        fluorescent_contours, _ = cv2.findContours(fluorescent_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        non_fluorescent_contours, _ = cv2.findContours(non_fluorescent_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        fluorescent_count = len(fluorescent_contours)
        non_fluorescent_count = len(non_fluorescent_contours)

        # 결과 이미지에 형광/비형광 콜로니 표시
        result_image = image_np.copy()

        for idx, contour in enumerate(fluorescent_contours):
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(result_image, f'F{idx + 1}', (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.drawContours(result_image, [contour], -1, (255, 0, 0), 2)

        for idx, contour in enumerate(non_fluorescent_contours):
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(result_image, f'NF{idx + 1}', (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.drawContours(result_image, [contour], -1, (0, 0, 255), 2)

        # 파일 저장
        original_image_path = os.path.join(TEMP_FOLDER, f"{image_file.filename}_original.jpg")
        result_image_path = os.path.join(TEMP_FOLDER, f"{image_file.filename}_result.jpg")

        Image.fromarray(image_np).save(original_image_path)
        Image.fromarray(result_image).save(result_image_path)

        colony_counts.append({
            "filename": image_file.filename,
            "fluorescent_count": fluorescent_count,
            "non_fluorescent_count": non_fluorescent_count
        })

        images.append({
            "original": f"{image_file.filename}_original.jpg",
            "result": f"{image_file.filename}_result.jpg"
        })

    return jsonify({
        "colony_counts": colony_counts,
        "images": images
    })

@app.route('/get_image/<filename>')
def get_image(filename):
    return send_from_directory(TEMP_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
