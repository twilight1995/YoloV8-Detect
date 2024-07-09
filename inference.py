import time

import cv2
from ultralytics import YOLO
from cv2 import getTickCount, getTickFrequency


def predict_camera(model):
    # 获取摄像头内容，参数 0 表示使用默认的摄像头
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        loop_start = getTickCount()
        success, frame = cap.read()  # 读取摄像头的一帧图像

        if success:
            # 对当前帧进行目标检测并显示结果，返回置信度大于conf的类别，verbose控制是否打印信息
            results = model.predict(source=frame, conf=0.4, verbose=False)

        annotated_frame = results[0].plot()
        # 中间放自己的显示程序
        loop_time = getTickCount() - loop_start
        total_time = loop_time / (getTickFrequency())
        FPS = int(1 / total_time)
        # 在图像左上角添加FPS文本
        fps_text = f"FPS: {FPS:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        text_color = (0, 0, 255)  # 红色
        text_position = (10, 30)  # 左上角位置

        cv2.putText(annotated_frame, fps_text, text_position, font, font_scale, text_color, font_thickness)
        cv2.imshow('img', annotated_frame)
        # 通过按下 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()  # 释放摄像头资源
    cv2.destroyAllWindows()  # 关闭OpenCV窗口


def predict_image(model, img_path):
    img = cv2.imread(img_path)
    # 对当前帧进行目标检测并显示结果，返回置信度大于conf的类别，verbose控制是否打印信息
    results = model.predict(source=img, conf=0.4, device='cpu', verbose=False)

    annotated_frame = results[0].plot()  # 注释后的可视化图片，需要的话可以直接返回

    print("bbox info")
    print("*" * 80)
    orig_img = results[0].orig_img  # 原始图像
    detect_info = results[0].boxes  # 检测结果，包括类别索引，包围框的信息等
    bboxes_data = detect_info.data  # 包围框信息
    # print(bboxes.data)
    for data in bboxes_data:
        # 每个data输出6个数，前4个是bbox，第5个是置信度，第6个是类别索引，类别信息再dataset.yaml中
        print(data.cpu().detach().numpy())

    return annotated_frame


if __name__ == '__main__':
    start_time = time.time()
    # 加载 YOLOv8 模型
    model = YOLO("weights/best.pt")  # 模型加载路径
    print(f"time cost: {time.time() - start_time}")
    img_path = "./data/test1.jpg"
    for i in range(100):
        annotated_frame = predict_image(model, img_path)
        print(f"time cost: {time.time() - start_time}")
        cv2.imshow('img', annotated_frame)
        cv2.waitKey(0)
        start_time = time.time()

