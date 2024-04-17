application_name = 'Image Captioning'
# pyqt packages
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox, QFileDialog


import sys
import numpy as np
import pickle
from torchvision import transforms
import torch
from PIL import Image

from model import Seq2SeqAttentionCNN
from qt_main import Ui_Application
from main import BidirectionalMap, tensorToTokens, PadToSquare, GPU_Device


def show_message(parent, title, message, icon=QMessageBox.Warning):
        msg_box = QMessageBox(icon=icon, text=message)
        msg_box.setWindowIcon(parent.windowIcon())
        msg_box.setWindowTitle(title)
        msg_box.setStyleSheet(parent.styleSheet() + 'color:white} QPushButton{min-width: 80px; min-height: 20px; color:white; \
                              background-color: rgb(91, 99, 120); border: 2px solid black; border-radius: 6px;}')
        msg_box.exec()
        
        
        
class QT_Action(Ui_Application, QMainWindow):
    def __init__(self):
        # system variable
        super(QT_Action, self).__init__()
        self.setupUi(self)
        self.retranslateUi(self)
        self.setWindowTitle(application_name) # set the title
        
        # runtime variable
        self.image = None
        self.model = None
        self.weight = None
        self.activated_features = None
        self.transform = None
        self.seq_len = 64 # maximum output length
        with open('English.pkl', 'rb') as f:
            self.vocab = pickle.load(f)
        
        # load the model
        self.load_model_action()
        
        
    # linking all button/textbox with actions    
    def link_commands(self,):
        self.toolButton_import.clicked.connect(self.import_action)
        self.comboBox_model.activated.connect(self.load_model_action)
        self.toolButton_process.clicked.connect(self.process_action)
        
    
    # choosing between models
    def load_model_action(self,):
        self.model_name = self.comboBox_model.currentText()
        
        # load the model
        if self.model_name == 'AttentionCNN':
            output_dim = len(self.vocab)
            sos_token = self.vocab.get_value('<sos>')
            eos_token = self.vocab.get_value('<eos>')
            
            # load the model architechture
            self.model = Seq2SeqAttentionCNN(output_dim, sos_token, eos_token)
            
            # loading the training model weights
            self.model.load_state_dict(torch.load(f'Seq2Seq{self.model_name}.pth'))
            
            # input image transform
            self.transform = transforms.Compose([
                PadToSquare(), 
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            
        self.model = self.model.to(GPU_Device())
        
        self.model.eval() # Set model to evaluation mode
        
        
        
    
    # clicking the import button action
    def import_action(self,):
        # show an "Open" dialog box and return the path to the selected file
        filename, _ = QFileDialog.getOpenFileName(None, "Select file", options=QFileDialog.Options())
        self.lineEdit_import.setText(filename)
        
        # didn't select any files
        if filename is None or filename == '': 
            return
    
        # selected .oct or .octa files
        if filename.endswith('.jpg'):
            self.image = Image.open(filename) 
            self.lineEdit_import.setText(filename)
            #X = [transform(img)]
            self.update_display()
        
        # selected the wrong file format
        else:
            show_message(self, title='Load Error', message='Available file format: .jpg')
            self.import_action()
        
        
    def update_display(self):
        if not self.image is None:
            data = self.transform(self.image).numpy()
            data = data.transpose(1,2,0)
            data = (data*255).astype(np.uint8)
            height, width, channels = data.shape
            q_image = QImage(data.tobytes(), width, height, width*channels, QImage.Format_RGB888)  # Create QImage
            qpixmap = QPixmap.fromImage(q_image)  # Convert QImage to QPixmap
            self.label_image.setPixmap(qpixmap)
            
            
    def process_action(self):
        if self.image is None:
            show_message(self, title='Process Error', message='Please load an image first')
            return
        
        # apply the transform
        data = self.transform(self.image)
        
        # move data to GPU
        data = data.to(GPU_Device())
        
        # model inference
        with torch.no_grad():  # Disable gradient calculation
            caption_token = self.model.inference(data, self.seq_len)
        
        # convert to tensor
        out_sentence = tensorToTokens(self.vocab, caption_token)
                
        # print out the output sentence
        out_sentence = ' '.join(out_sentence)
            
        # Display the result in label_heatmap
        self.textEdit_caption.setPlainText(out_sentence)
    
    
def main():
    app = QApplication(sys.argv)
    action = QT_Action()
    action.link_commands()
    action.show()
    sys.exit(app.exec_())
    
    
if __name__ == '__main__':
    main()