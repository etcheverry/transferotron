import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QWidget, QAction, QTabWidget,QVBoxLayout
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtMultimedia import QSound

import numpy as np

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import librosa
import cmath
import os

import random

def remove_extension(string):
    return os.path.splitext(os.path.split(string)[1])[0]

class App(QMainWindow):
 
    def __init__(self):
        super().__init__()
        self.title = 'Transferotron'
        self.left = 0
        self.top = 0
        self.width = 800
        self.height = 500
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
 
        self.table_widget = MyTableWidget(self)
        self.setCentralWidget(self.table_widget)
 
        self.showMaximized()
 
class MyTableWidget(QWidget):

    def valueChange(self):
        self.iteration = self.it_spin.value()
                
    def __init__(self, parent):   
        super(QWidget, self).__init__(parent)
        self.layout = QVBoxLayout(self)
        
        self.iteration = 9
        self.generation_type = 'generated'
        self.it_spin = QSpinBox()
        self.it_spin.setMinimum(0)
        self.it_spin.setMaximum(9)
        self.it_spin.setValue(9)
        self.generation = QLineEdit()
        self.generation.setText('generated')
        self.phase = QCheckBox("Display phase")
        
        # Initialize tab screen
        self.tabs = QTabWidget()
        self.tab1 = QWidget()	
        self.tab2 = QWidget()
        self.tabs.resize(300,200) 
 
        # Add tabs
        self.tabs.addTab(self.tab1,"Display")
        self.tabs.addTab(self.tab2,"Generate") 
        
        # Create first tab

        self.tab1.main = QHBoxLayout(self)
        self.tab1.layout = QVBoxLayout(self)
        self.tab1.layout.horizontal = QHBoxLayout(self)
        self.tab1.layout.vertical = QVBoxLayout(self)
        self.tab1.layout.vertical.addWidget(self.it_spin)
        self.tab1.layout.vertical.addWidget(self.phase)
        self.tab1.layout.vertical.addWidget(self.generation)
        self.tab1.button = QPushButton('Select Folder', self)
        self.tab1.button.clicked.connect(self.showDialog)
        self.tab1.layout.vertical.addWidget(self.tab1.button)
        self.tab1.layout.addLayout(self.tab1.layout.horizontal)
        
        #Create left panel
        self.tab1.left = QVBoxLayout(self)
        self.tab1.left.fileModel = QFileSystemModel()
        self.tab1.left.listView = QListView()
        self.tab1.left.selected = 'nofile'
        self.tab1.left.addWidget(self.tab1.left.listView)
        self.tab1.layout.vertical.addLayout(self.tab1.left)
        #Create right panel
        self.tab1.right = QVBoxLayout(self)
        self.tab1.right.fileModel = QFileSystemModel()
        self.tab1.right.listView = QListView()
        self.tab1.right.selected = 'nofile'
        self.tab1.right.addWidget(self.tab1.right.listView)
        self.tab1.layout.vertical.addLayout(self.tab1.right)

        self.tab1.layout.horizontal.addLayout(self.tab1.layout.vertical)

        #Play button
        self.tab1.play_content = QPushButton('Play', self)
        self.tab1.play_content.clicked.connect(self.play_sound_content)
        #Play button
        self.tab1.play_style = QPushButton('Play', self)
        self.tab1.play_style.clicked.connect(self.play_sound_style)
        #Play button
        self.tab1.play_result = QPushButton('Play', self)
        self.tab1.play_result.clicked.connect(self.play_sound_result)

        #Create View panel
        self.tab1.view = QVBoxLayout(self)
        self.tab1.view.content = PlotCanvas(self, width=5, height=2, title='content')
        self.tab1.view.content_snd = PlotRawSound(self, width=5, height=1, amplitude_canvas=self.tab1.view.content)
        
        self.tab1.view.style = PlotCanvas(self, width=5, height=2, title='style')
        self.tab1.view.style_snd = PlotRawSound(self, width=5, height=1, amplitude_canvas=self.tab1.view.style)
        
        self.tab1.view.result = PlotCanvas(self, width=5, height=2, title='result')
        self.tab1.view.result_snd = PlotRawSound(self, width=5, height=1, amplitude_canvas=self.tab1.view.result)
        
        self.tab1.view.layoutContent = QHBoxLayout(self)
        self.tab1.view.layoutContent.addWidget(self.tab1.play_content)
        self.tab1.view.layoutContent.addWidget(self.tab1.view.content)
        
        self.tab1.view.layoutStyle = QHBoxLayout(self)
        self.tab1.view.layoutStyle.addWidget(self.tab1.play_style)
        self.tab1.view.layoutStyle.addWidget(self.tab1.view.style)
        
        self.tab1.view.layoutResult = QHBoxLayout(self)
        self.tab1.view.layoutResult.addWidget(self.tab1.play_result)
        self.tab1.view.layoutResult.addWidget(self.tab1.view.result)
        
        self.tab1.view.addLayout(self.tab1.view.layoutContent)
        self.tab1.view.addWidget(self.tab1.view.content_snd)
        
        self.tab1.view.addLayout(self.tab1.view.layoutStyle)
        self.tab1.view.addWidget(self.tab1.view.style_snd)
        
        self.tab1.view.addLayout(self.tab1.view.layoutResult)
        self.tab1.view.addWidget(self.tab1.view.result_snd)


        self.tab1.layout.horizontal.addLayout(self.tab1.view, 2)

        # Add panels
        self.tab1.main.addLayout(self.tab1.layout)
        self.tab1.setLayout(self.tab1.main)
 
        # Add tabs to widget        
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)

    def play_sound_content(self):
        if(self.tab1.left.selected != 'nofile'):
            QSound.play(self.fname + '/' + self.tab1.left.selected)

    def play_sound_style(self):
        if(self.tab1.right.selected != 'nofile'):
            QSound.play(self.fname + '/' + self.tab1.right.selected)

    def play_sound_result(self):
        combined = remove_extension(self.tab1.left.selected) + '_' + remove_extension(self.tab1.right.selected)
        path = self.fname +  '/' + self.generation.text() +  '/' + combined + '/' + combined + '_at_iteration_' + str(self.iteration) + '.wav'
        if(os.path.isfile(path)):
            QSound.play(path)

    def showDialog(self):
        self.fname = QFileDialog.getExistingDirectory(self, 'Select Directory', '/autofs/netapp/travail/nicetcheverry/stage/transferotron/outs/')
        if(self.fname != ''):
            self.tab1.left.fileModel.setFilter(QDir.Files)
            self.tab1.left.fileModel.setRootPath(self.fname)
            self.tab1.left.listView.setModel(self.tab1.left.fileModel)
            self.tab1.left.listView.setRootIndex(self.tab1.left.fileModel.index(self.fname))
            self.tab1.left.listView.clicked.connect(self.on_click_left)
            
            self.tab1.right.fileModel.setFilter(QDir.Files)
            self.tab1.right.fileModel.setRootPath(self.fname)
            self.tab1.right.listView.setModel(self.tab1.right.fileModel)
            self.tab1.right.listView.setRootIndex(self.tab1.right.fileModel.index(self.fname))
            self.tab1.right.listView.clicked.connect(self.on_click_right)
 
    @pyqtSlot("QModelIndex")
    def on_click_left(self, index):
        self.tab1.left.selected = index.data()
        self.tab1.view.content.plot(self.fname + '/' + self.tab1.left.selected)
        self.tab1.view.content_snd.plot(self.fname + '/' + self.tab1.left.selected)
        if(self.tab1.right.selected != 'nofile'):
            combined = remove_extension(self.tab1.left.selected) + '_' + remove_extension(self.tab1.right.selected)
            path = self.fname +  '/' + self.generation.text() +  '/' + combined + '/' + combined + '_at_iteration_' + str(self.iteration) + '.wav'
            if(os.path.isfile(path)):
                self.tab1.view.result.plot(path)
                self.tab1.view.result_snd.plot(path)
            else:
                self.tab1.view.result.plot('nofile')
                self.tab1.view.result_snd.plot('nofile')
        
    @pyqtSlot("QModelIndex")
    def on_click_right(self, index):
        self.tab1.right.selected = index.data()
        self.tab1.view.style.plot(self.fname + '/' + self.tab1.right.selected)
        self.tab1.view.style_snd.plot(self.fname + '/' + self.tab1.right.selected)
        if(self.tab1.left.selected != 'nofile'):
            combined = remove_extension(self.tab1.left.selected) + '_' + remove_extension(self.tab1.right.selected)
            path = self.fname + '/' + self.generation.text() +  '/' + combined + '/' + combined + '_at_iteration_' + str(self.iteration) + '.wav'
            if(os.path.isfile(path)):
                self.tab1.view.result.plot(path)
                self.tab1.view.result_snd.plot(path)
            else:
                self.tab1.view.result.plot('nofile')
                self.tab1.view.result_snd.plot('nofile')

            
class PlotCanvas(FigureCanvas):
 
    def __init__(self, parent=None, width=5, height=4, dpi=100, title='No title', filename='nofile'):
        self.title = title
        self.current_time = 0
        self.parent=parent
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.plot('nofile')
 
 
    def plot(self, filename):
        self.y = []
        if(filename != 'nofile'):
            print('Loading : ' + filename)
            sound, sr = librosa.load(filename, sr=None)
            print('Loaded')
            stft = librosa.stft(sound)
            print('Fouriered')
            print(stft.shape)
            self.data = []
            for i in range(stft.shape[0]):
                if(self.parent.phase.isChecked()):
                    self.data.append(cmath.polar(stft[i][self.current_time])[1])
                else:
                    self.data.append(cmath.polar(stft[i][self.current_time])[0])
            print(len(self.data))
            self.axes.clear()
            self.ax = self.figure.add_subplot(111)
            self.ax.set_ylim(0, max(self.data)+5)
            freq = librosa.fft_frequencies()
            self.ax.set_xlim(0, max(freq))
            self.ax.set_xlabel(u'frequency (Hz)', fontsize=6)
            ticks = np.arange(0, max(freq), 500)
            small_ticks = np.arange(0, max(freq), 100)
            self.axes.set_xticks(ticks)
            self.axes.set_xticks(small_ticks, True)
            self.ax.plot(freq, self.data)
            self.ax.set_title(self.title)
            self.draw()
        else:
            self.axes.clear()
            self.ax = self.figure.add_subplot(111)
            self.ax.set_ylim(0, 1)
            self.ax.set_xlim(0, 1)
            self.axes.text(0.2,0.2,'Not Found')
            self.draw()
        
def onclick(event):
    print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %('double' if event.dblclick else 'single', event.button, event.x, event.y, event.xdata, event.ydata))
    event.canvas.parent.tab1.view.content_snd.set_line(event.xdata)
    event.canvas.parent.tab1.view.style_snd.set_line(event.xdata)
    event.canvas.parent.tab1.view.result_snd.set_line(event.xdata)


class PlotRawSound(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100, title='No title', filename='nofile', amplitude_canvas='None'):
        self.amplitude_canvas = amplitude_canvas
        self.title = title
        self.filename = filename
        self.linepos = 20
        self.parent = parent
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        cid = self.mpl_connect('button_press_event', onclick)
        self.plot('nofile')
 
 
    def plot(self, filename):
        self.y = []
        self.filename = filename
        if(filename != 'nofile'):
            print('Loading : ' + filename)
            sound, sr = librosa.load(filename, sr=None)
            print('Loaded')
            self.axes.clear()
            self.ax = self.figure.add_subplot(111)
            self.ax.set_ylim(-1, 1)
            self.ax.set_xlim(0, len(sound))
            self.ax.vlines(x=self.linepos, ymin=-1,ymax=1, color="red",linewidth=2, zorder=3)
            self.ax.plot(sound)
            self.draw()
        else:
            self.axes.clear()
            self.ax = self.figure.add_subplot(111)
            self.ax.set_ylim(0, 1)
            self.ax.set_xlim(0, 1)
            self.axes.text(0.2,0.2,'Not Found')
            self.draw()
            
    def set_line(self, x):
        self.linepos = x
        self.plot(self.filename)
        self.amplitude_canvas.current_time = int(x//(2048/4))
        self.amplitude_canvas.plot(self.filename)
        
    
    
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
