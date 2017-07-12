import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
#from scipy.ndimage.interpolation import rotate
from scipy.misc import imresize
import tensorflow as tf
#import Augmentation


"""
HVSMR 2016: MICCAI
Whole-Heart and Great Vessel Segmentation from 
3D Cardiovascular MRI in Congenital Heart Disease

Class Label:
0 = Background
1 = 
2 = 
"""

def showImage(data, segment, slice=55, cmap_='hot', vmin=None, vmax=None):
    # segment:
    # black=0, orange=liver, white=tumor
    plt.figure(figsize=(8,8))
    plt.subplot(1,2,1)
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    plt.imshow(data[:,:,slice], cmap=cmap_, vmin=vmin, vmax=vmax)
    plt.title('Img Dim {0}, slice {1}'.format(data.shape, slice))
    plt.subplot(1,2,2)
    plt.imshow(segment[:,:,slice], vmin=0, vmax=2, cmap='hot')

def frequencyTable(label):
    unique_ = np.unique(label)
    pixels_ = float(len(label.flatten()))
    table_ = [(i, len(label[label == i])) for i in unique_]
    print( table_ )
    print( [(i[0],str(i[1]/float(pixels_))[:6]+'%') for i in table_] )

def locateCenter(label):
    coord = np.where(label==2)
    print('center : {}'.format(np.average(coord,1)))


#path = "axial_crop"
#path = "axial_full"
#path = "sa_crop"
#
#index = 7
#segmentFile = "./datasetHVSMR16Heart/"+path+"/GroundTruth/training_"+path+"_pat"
#volumeFile = "./datasetHVSMR16Heart/"+path+"/TrainingDataset/training_"+path+"_pat"
#segment = sitk.GetArrayFromImage(sitk.ReadImage(segmentFile+str(index)+'-label.nii.gz'))
#data = sitk.GetArrayFromImage(sitk.ReadImage(volumeFile+str(index)+'.nii.gz'))
#segment.shape
#
#frequencyTable(segment)
#locateCenter(segment)
#
#slice = 65
#showImage(data,segment,slice, cmap_='CMRmap')
#showImage(data,segment,slice, cmap_='CMRmap', vmin=0 ,vmax=3000) # might need to do clipping
#
#path = "axial_crop"
#for index in range(10):
#    volumeFile = "./datasetHVSMR16Heart/"+path+"/TrainingDataset/training_"+path+"_pat"
#    data = sitk.GetArrayFromImage(sitk.ReadImage(volumeFile+str(index)+'.nii.gz'))
#    print(data.shape)


class HVSMRdataset():
    filename = {"axial_crop":'training_axial_crop_pat{}.nii.gz',
    "axial_full":'training_axial_full_pat{}.nii.gz',
    "sa_crop":'training_sa_crop_pat{}.nii.gz'}    
    
    listFolder = []
    listTrain = []
    listValid = []
    __trainIndex = 0
    __validIndex = 0
    
    def __init__(self, filepath="./datasetHVSMR16Heart" , type='axial_crop'):
        assert os.path.exists(os.path.join(filepath,type)), "no such path directory"
        self.filepath = os.path.join(filepath,type)
        self.filetype = type
        self.InitDataset()
        assert self.AbleToRetrieveData(), "not able to retrieve data from path."
        
    def SetPath(self, filepath):
        assert os.path.exists(filepath), "no such path directory"
        self.filepath = filepath
      
    def AbleToRetrieveData(self, idx=''):
        if os.path.exists(os.path.join(self.filepath,'TrainingDataset',idx)):
            return True
        else: 
            return False
            
    def InitDataset(self,splitRatio=1.0, shuffle=False, seed=0):
        self.listFolder = range(10)
        self.listTrain = range(10)
        self.listValid = range(10)
    
    def __pltImage(self, data, segment, index=55):
        cmap_ = 'hot'
        # segment:
        # black=0, orange=liver, white=tumor
        plt.figure(figsize=(8,8))
        plt.subplot(1,2,1)
        plt.imshow(data[index], cmap=cmap_)
        plt.title('Image Dim: {}'.format(data.shape))
        plt.subplot(1,2,2)
        plt.imshow(segment[index], vmin=0, vmax=2, cmap=cmap_)
        
    def ShowSegmentation(self, idx=4, slice=55):
        assert self.AbleToRetrieveData(''), "unable to retrieve image files"
        segmentPath = os.path.join(self.filepath,'GroundTruth',self.filename[self.filetype])
        segmentFile = segmentPath.format(str(idx)+'-label')
        volumePath = os.path.join(self.filepath,'TrainingDataset',self.filename[self.filetype])
        volumeFile = volumePath.format(idx)
        segment = sitk.GetArrayFromImage(sitk.ReadImage(segmentFile))
        data = sitk.GetArrayFromImage(sitk.ReadImage(volumeFile))
        self.__pltImage(data, segment, index=slice)
    
    def padding(self,x,dim=(181,239,165)):
        dataShape = x.shape
        d1 = int(np.ceil((dim[0]-dataShape[0])/2.0))
        d2 = int(np.floor((dim[0]-dataShape[0])/2.0))
        w1 = int(np.ceil((dim[1]-dataShape[1])/2.0))
        w2 = int(np.floor((dim[1]-dataShape[1])/2.0))
        h1 = int(np.ceil((dim[2]-dataShape[2])/2.0))
        h2 = int(np.floor((dim[2]-dataShape[2])/2.0))
        return np.pad(x,[[d1,d2],[w1,w2],[h1,h2]],'constant')    
    
    def resize(self,x,dim=(100,100,100)):
        return imresize(x, dim, interp='bilinear', mode=None)
    
    def __nextBatch(self,batchSize,dataset='train'):
        assert len(self.listFolder) > 0, 'Please initialize dataset first, by calling InitDataset()'
        if dataset == 'train':
            dataFolder = self.listTrain
            batchIndex = self.__trainIndex
        else:
            dataFolder = self.listValid
            batchIndex = self.__validIndex
        length = len(dataFolder)
        batchEnd = batchIndex + batchSize
        start_ = np.mod(batchIndex,length)
        end_ = np.mod(batchEnd,length)
        if batchSize == length:
            batchPath_ = dataFolder
        elif end_ < (start_):
            batchPath_ = dataFolder[start_:]+dataFolder[:(end_)]
        else:
            batchPath_ = dataFolder[start_:end_]
        if dataset == 'train':
            self.__trainIndex = batchEnd
        else:
            self.__validIndex = batchEnd
        return batchPath_
    
    def NextBatch3D(self,batchSize,dataset='train'):
        batchPath_ = self.__nextBatch(batchSize,dataset)
        segmentPath = os.path.join(self.filepath,'GroundTruth',self.filename[self.filetype])
        segmentPath = [segmentPath.format(str(i)+'-label') for i in batchPath_]
        volumePath = os.path.join(self.filepath,'TrainingDataset',self.filename[self.filetype])
        volumePath = [volumePath.format(i) for i in batchPath_]
        #
        print('fetching '+dataset+' rawdata from drive')
        # maxValue = 3300.0
        maxValue = 1.0
        segmentData_ = [self.padding(sitk.GetArrayFromImage(sitk.ReadImage(i)))/maxValue for i in segmentPath]  
        volumeData_ = [self.padding(sitk.GetArrayFromImage(sitk.ReadImage(i)))/maxValue for i in volumePath]  
        #segmentData_ = [self.resize(sitk.GetArrayFromImage(sitk.ReadImage(i)))/maxValue for i in segmentPath]  
        #volumeData_ = [self.resize(sitk.GetArrayFromImage(sitk.ReadImage(i)))/maxValue for i in volumePath] 
        #segmentData_ = [sitk.GetArrayFromImage(sitk.ReadImage(i))/maxValue for i in segmentPath]  
        #volumeData_ = [sitk.GetArrayFromImage(sitk.ReadImage(i))/maxValue for i in volumePath] 
        #segmentData_ = np.array(segmentData_)
        #volumeData_ = np.array(volumeData_)
        segmentData_ = np.array([i.reshape(i.shape+(1,)) for i in segmentData_])
        volumeData_ = np.array([i.reshape(i.shape+(1,)) for i in volumeData_])
        return volumeData_, segmentData_

    
    
    
    
    
    




