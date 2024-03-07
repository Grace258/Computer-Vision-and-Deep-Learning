import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, QFileDialog, QComboBox, QScrollArea, QVBoxLayout, QHBoxLayout, QLineEdit, QGraphicsScene, QGraphicsPixmapItem, QGraphicsView, QGraphicsRectItem)
from PyQt5.QtGui import QFont, QImage, QPixmap, QPainter, QColor, QPen, QPainterPath
from PyQt5.QtCore import Qt, QRectF, QTimer, QPoint, QBuffer
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import subprocess
from VGG19 import VGG19BN
from ResNet50 import CustomResNet50
from torchsummary import summary
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import PIL.Image as Image
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from skimage import io, color

class DrawingBoard(QGraphicsView):
    def __init__(self, parent=None):
        super(DrawingBoard, self).__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setSceneRect(0, 0, 1200, 600)       

        self.pixmap_item = QGraphicsPixmapItem()
        self.scene().addItem(self.pixmap_item)

        self.painter = QPainter()
        self.last_point = QPoint()

        # Set pen color to white
        self.pen = QPen(Qt.white, 40, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)  #筆調粗

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.last_point = event.pos()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            pixmap = QPixmap(self.pixmap_item.pixmap())  # Create a new pixmap
            self.painter.begin(pixmap)
            self.painter.setPen(self.pen)
            self.painter.drawLine(self.last_point, event.pos())
            self.painter.end()

            self.pixmap_item.setPixmap(pixmap)
            self.last_point = event.pos()

    def resizeEvent(self, event):
        self.pixmap_item.setPixmap(QPixmap(self.width(), self.height()))

    def get_image_data(self):
        gray_img = self.pixmap_item.pixmap().toImage().convertToFormat(QImage.Format_Grayscale8)
        
        width, height = gray_img.width(), gray_img.height()
        ptr = gray_img.bits()
        ptr.setsize(width * height)
        img_array = np.frombuffer(ptr, dtype=np.uint8).reshape((height, width))

        img_array = 1.0 - (img_array/255)
        img_array = (img_array - 0.1307) / 0.3081
        
        img_resized = cv2.resize(img_array, (32, 32))        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        img_tensor = torch.FloatTensor(img_resized).to(device)
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)

        return img_tensor



class MyWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.image_path = None
        self.video_path = None
        self.cap = None
        self.points = None
        self.tracked_points = [] 
        self.drawing_board = None       
        self.classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']             
        
        self.initUI()

    def initUI(self):
        self.setWindowTitle('MainWindow-cvdlhw1')
        self.setGeometry(50, 50, 2100, 1150)                         
        
        self.label1 = QLabel('1. Background Substraction', self)
        self.label1.move(180, 50)
        self.label1.setFont(QFont('Arial', 12))

        self.label2 = QLabel('2. Optical Flow', self)
        self.label2.move(180, 180)
        self.label2.setFont(QFont('Arial', 12)) 

        self.label3 = QLabel('3. PCA', self)
        self.label3.move(180, 350)
        self.label3.setFont(QFont('Arial', 12))

        self.label4 = QLabel('4. MNIST Classifier Using VGG19', self)
        self.label4.move(530, 50)
        self.label4.setFont(QFont('Arial', 12)) 

        self.label5 = QLabel('5. ResNet50', self)
        self.label5.move(530, 800)
        self.label5.setFont(QFont('Arial', 12))  

        self.label6 = QLabel(self)
        self.label6.setGeometry(800, 600, 224, 224) 
        self.label6.setFont(QFont('Arial', 12)) 

        self.label7 = QLabel(self)
        self.label7.setGeometry(900, 850, 224, 224) 

        self.label8 = QLabel(self)
        self.label8.setGeometry(900, 980, 250, 250)
        self.label8.setFont(QFont('Arial', 12))         

        
        self.button1 = QPushButton('Load Image', self) 
        self.button1.move(30, 100)
        self.button1.setFixedSize(120, 40)
        self.button1.clicked.connect(self.LoadImage)

        self.button2 = QPushButton('Load Video', self) 
        self.button2.move(30, 170)
        self.button2.setFixedSize(120, 40)
        self.button2.clicked.connect(self.LoadVideo)

        self.button3 = QPushButton('1. Background Substraction', self) 
        self.button3.move(200, 100)
        self.button3.setFixedSize(210, 40)
        self.button3.clicked.connect(self.BackgroundSubstraction)

        self.button4 = QPushButton('2.1 Preprocessing', self) 
        self.button4.move(200, 230)
        self.button4.setFixedSize(210, 40)
        self.button4.clicked.connect(self.Preprocessing)

        self.button5 = QPushButton('2.2 Video tracking', self) 
        self.button5.move(200, 280)
        self.button5.setFixedSize(210, 40)
        self.button5.clicked.connect(self.VideoTracking)

        self.button6 = QPushButton('3. Dimension Reduction', self) 
        self.button6.move(200, 400)
        self.button6.setFixedSize(210, 40)
        self.button6.clicked.connect(self.DimensionReduction)

        self.button7 = QPushButton('4.1 Show Model Structure', self) 
        self.button7.move(550, 100)
        self.button7.setFixedSize(210, 40)
        self.button7.clicked.connect(self.ShowModelStructure)

        self.button8 = QPushButton('4.2 Show Accuracy and Loss', self) 
        self.button8.move(550, 150)
        self.button8.setFixedSize(210, 40)
        self.button8.clicked.connect(self.ShowAccuracyAndLoss)

        self.button9 = QPushButton('4.3 Predict', self) 
        self.button9.move(550, 200)
        self.button9.setFixedSize(210, 40)
        self.button9.clicked.connect(self.Predict)            

        self.drawing_board = DrawingBoard(self)  
        self.drawing_board.move(800, 100)
        self.drawing_board.setFixedSize(1200, 600)   

        self.button10 = QPushButton('4.4 Reset', self) 
        self.button10.move(550, 250)
        self.button10.setFixedSize(210, 40)
        self.button10.clicked.connect(self.drawing_board.resizeEvent)

        self.button11 = QPushButton('Load Images', self) 
        self.button11.move(550, 850)
        self.button11.setFixedSize(210, 40)
        self.button11.clicked.connect(self.LoadImages)

        self.button12 = QPushButton('5.1 Show Images', self) 
        self.button12.move(550, 900)
        self.button12.setFixedSize(210, 40)
        self.button12.clicked.connect(self.ShowImages)

        self.button13 = QPushButton('5.2 Show Model Structure', self) 
        self.button13.move(550, 950)
        self.button13.setFixedSize(210, 40)
        self.button13.clicked.connect(self.ShowModelStructure_ResNet)

        self.button14 = QPushButton('5.3 Show Comparison', self) 
        self.button14.move(550, 1000)
        self.button14.setFixedSize(210, 40)
        self.button14.clicked.connect(self.ShowComparison)

        self.button15 = QPushButton('5.4 Inference', self) 
        self.button15.move(550, 1050)
        self.button15.setFixedSize(210, 40)
        self.button15.clicked.connect(self.Inference_ResNet)
                
        self.show()

    def LoadImage(self):
        self.image_path, filetype = QFileDialog.getOpenFileName(self, "Open file", "./")
        print("image_path = ", self.image_path) 

    def LoadVideo(self):
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setOption(QFileDialog.DontUseNativeDialog, False)
        file_dialog.setNameFilter('MP4 (*.mp4)')
        self.video_path, _ = file_dialog.getOpenFileNames(self, 'Select MP4', '/', 'MP4 (*.mp4)')
        self.cap = cv2.VideoCapture(self.video_path[0])
        print("video_path = ", self.video_path) 

    def BackgroundSubstraction(self):        
        if self.cap is None:
            print("No file is choosen. Please choose one.")
            return        
        # 創建背景分離器
        history = 500
        dist2Threshold = 400
        bg_subtractor = cv2.createBackgroundSubtractorKNN(history, dist2Threshold, detectShadows=True)
        # 跳過前4幀
        for _ in range(4):
            _, _ = self.cap.read()

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            # 對幀進行模糊處理
            blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
            # 獲取背景遮罩
            mask = bg_subtractor.apply(blurred_frame)
            # 生成僅包含運動對象的幀
            result_frame = cv2.bitwise_and(blurred_frame, blurred_frame, mask=mask)
            # 顯示結果
            cv2.imshow('RGB frame', frame)
            cv2.imshow('Foreground mask', mask)
            cv2.imshow('Result', result_frame)
            # 如果按下 'q' 鍵，則退出循環
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        # 釋放視頻捕獲對象並關閉窗口
        self.cap.release()
        cv2.destroyAllWindows()
    
    def Preprocessing(self):
        
        if self.cap is None:
            print("No file is choosen. Please choose one.")
            return
        ret, frame = self.cap.read()
        # 將幀轉換為灰度
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 檢測娃娃鼻子底部的點
        corners = cv2.goodFeaturesToTrack(gray_frame, maxCorners=1, qualityLevel=0.3, minDistance=7, blockSize=7)

        if corners is not None:
            self.points = corners.reshape(-1, 2)  # 將形狀改為 (1, 2)
            x, y = map(int, self.points[0])
            cv2.line(frame, (x - 10, y), (x + 10, y), color = (0,0,255), thickness = 2)
            cv2.line(frame, (x, y - 10), (x, y + 10), color = (0,0,255), thickness = 2) 
        # 顯示帶有檢測點的幀
        cv2.imshow("Optical flow", frame)

    def VideoTracking(self):
        if self.cap is None or self.points is None:
            print("Please choose a file and check the points.")
            return

        self.points = self.points.astype(np.float32)      

        ret, prev_frame = self.cap.read()
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)        

        while True:
            ret, next_frame = self.cap.read()
            if not ret:
                break
            next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
            next_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, self.points, None)            
            
            if next_pts is not None:
                status = status.flatten()
                good_pts = next_pts[status == 1]
            else:
                break

            for pt in good_pts:
                x, y = pt.ravel()
                cv2.circle(next_frame, (int(x), int(y)), 5, (0, 100, 255), -1) 

            self.tracked_points.extend(good_pts)

            if len(self.tracked_points) > 1:
                cv2.polylines(next_frame, [np.int32(self.tracked_points)], isClosed=False, color=(0, 100, 255), thickness=2)
            
            cv2.imshow('Tracked Points', next_frame)
            prev_gray = next_gray.copy()
            self.points = good_pts
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
    
    def DimensionReduction(self):        
        image = io.imread(self.image_path)
        gray_image = color.rgb2gray(image)
        gray_image_normalized = gray_image / 255.0        

        width, height = gray_image.shape
        n_components = 50        
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(gray_image_normalized)        
        reconstructed_image = pca.inverse_transform(pca_result)
        reconstructed_image = reconstructed_image.reshape(width, height)        
        mse = mean_squared_error(gray_image_normalized, reconstructed_image)       
        
        print("mse: ", mse)                
        plt.subplot(1, 2, 1)
        plt.imshow(gray_image_normalized, cmap='gray')
        plt.title('Gray Scale Image')
        plt.subplot(1, 2, 2)
        plt.imshow(reconstructed_image, cmap='gray')
        plt.title(f'Reconstruction Image (n = {n_components})')
        plt.show()

    def ShowModelStructure(self):
        model = VGG19BN()
        model = model.to('cuda')
        summary(model, (1,32,32))

    def ShowAccuracyAndLoss(self):
        image = cv2.imread('C:/Users/yingc/Desktop/cvdl_hw2/training_plot_mnist.png')
        cv2.imshow('Training/Validating loss and accuracy', image)

    def run_Predict(self):
        # Get the drawn image data
        input_tensor = self.drawing_board.get_image_data()
        model_path = 'C:/Users/yingc/Desktop/cvdl_hw2/best_model_mnist.pth'
        model = torch.load(model_path)
        model.eval()        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)          

        output = model.forward(input_tensor)
        logits = output[0]
        confidence = F.softmax(logits, dim=1).cpu().detach().numpy()        
        confidence_tensor = torch.tensor(confidence)
        _, predicted_class = torch.max(confidence_tensor, dim=1)         

        self.label6.setText(f'predicted_class = {self.classes[predicted_class.item()]}')      

        print("predicted_class = ", self.classes[predicted_class.item()])              

        return confidence
    
    def Predict(self):
        confidence = self.run_Predict()
        self.show_histogram(confidence)

    def show_histogram(self, confidence):        
        confidence_values = confidence.flatten()
        plt.bar(self.classes, confidence_values)
        plt.xlabel('Class')
        plt.ylabel('Probability')
        plt.title('Histogram of Confidence')
        plt.show()

    def LoadImages(self):
        file_dialog = QFileDialog()
        self.image_path, _ = file_dialog.getOpenFileName(self, 'Open Image File', '', 'Images (*.png *.jpg *.bmp)')
        pixmap = QPixmap(self.image_path)
        pixmap = pixmap.scaledToWidth(224)  # Resize the image to 224x224
        self.label7.setPixmap(pixmap)             

    def ShowImages(self):
        inference_data_path = 'C:/Users/yingc/Desktop/cvdl_hw2/Dataset_Cvdl_Hw2_Q5/dataset/inference_dataset'
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        inference_dataset = ImageFolder(root=inference_data_path, transform=transform)        
        batch_size = 1  
        inference_data_loader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False) 

        images_to_show = []
        labels_to_show = set()
        for images, labels in inference_data_loader:
            label = labels.item()
            if label not in labels_to_show:
                images_to_show.append(images.squeeze(0))
                labels_to_show.add(label)
                if len(labels_to_show) == 2:
                    break

        plt.figure()        
        plt.subplot(1, 2, 1)
        plt.imshow(images_to_show[0].permute(1, 2, 0).numpy())  # 转换通道顺序
        plt.title("Cat")        
        plt.subplot(1, 2, 2)
        plt.imshow(images_to_show[1].permute(1, 2, 0).numpy())  # 转换通道顺序
        plt.title("Dog")
        plt.show()

    def ShowModelStructure_ResNet(self):
        model = CustomResNet50()
        model = model.to('cuda')
        summary(model, (3,224,224))
         

    def ShowComparison(self):
        image = cv2.imread("C:/Users/yingc/Desktop/cvdl_hw2/accuracy_comparison.png")
        cv2.imshow("Accuracy Comparison", image)

    def Inference_ResNet(self):
        model_path = 'C:/Users/yingc/Desktop/cvdl_hw2/best_RE_model.pth'
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load the model onto the correct device
        model = torch.load(model_path, map_location=device)
        model.eval()
        model = model.to(device)

        image = Image.open(self.image_path)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = transform(image).unsqueeze(0)

        # Move the input tensor to the same device as the model
        input_tensor = input_tensor.to(device)

        # Run inference
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)

        # Get the predicted class label based on a threshold (e.g., 0.5)
        predicted_class = torch.argmax(probabilities, dim=1).item()

        # Map the predicted class index to the actual class label
        class_labels = ["Cat", "Dog"]
        predicted_label = class_labels[predicted_class]

        # Display the predicted label
        self.label8.setText(f'Predicted class = {predicted_label}') 
        print("Predicted class = ", predicted_label)

  

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MyWidget()
    w.show()
    sys.exit(app.exec_())