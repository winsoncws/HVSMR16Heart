import numpy as np
import matplotlib.pyplot as plt  

def showScanImage2(slice, pic, label, pred, threshold=None, saturate=None, cmap_='CMRmap'):
    pic_ = np.copy(pic)
    label_ = np.copy(label)
    pred_ = np.copy(pred)
    if saturate is not None:
        print('saturate')
        temp = pred_[:,:,:,1]
        temp[temp > saturate] = 1
        temp = pred_[:,:,:,2]
        temp[temp > saturate] = 2
        #pic_[pic_ > saturate] = pic_.max()
    if threshold is not None:
        print('threshold')
        temp = pred_[:,:,:,0]
        temp[temp < threshold] = 0
        ###
    plt.figure(figsize=(8,8))
    plt.subplot(3,2,1)
    ####
    max_ = np.argmax(pred_,3)
    plt.imshow(max_[slice,:,:], cmap_, vmax=2)
    plt.title('Label Argmax')
    ####   
    #plt.imshow(pic_[slice,:,:,0]/6000.0*1, cmap_)
    plt.subplot(3,2,2)
    plt.imshow(pic_[slice,:,:,0]/2000.0*1, cmap_)    
    plt.title('Flair Image')
    plt.subplot(3,2,3)
    plt.imshow(label_[slice,:,:,0], cmap_, vmax=2)
    plt.title('Flair Label')
    plt.subplot(3,2,4)
    plt.imshow(pred_[slice,:,:,1], cmap_, vmax=2)   # change 1 to 0
    plt.title('Predicted Label')
    plt.subplot(3,2,5)
    plt.imshow(pred_[slice,:,:,2], cmap_, vmax=1)   # change 2 to 1
    plt.title('Predicted Others')
    plt.subplot(3,2,6)
    plt.imshow(pred_[slice,:,:,0], cmap_, vmax=1)
    plt.title('Predicted Noise')
    plt.tight_layout()

def showImage(data, segment, predict, slice=55, cmap_='hot', vmin=None, vmax=None):
    # segment:
    # black=0, orange=liver, white=tumor
    plt.figure(figsize=(8,8))
    plt.subplot(2,2,1)
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    plt.imshow(data[:,:,slice,0], cmap=cmap_, vmin=vmin, vmax=vmax)
    plt.title('Img Dim {0}, slice {1}'.format(data.shape, slice))
    plt.subplot(2,2,2)
    plt.imshow(segment[:,:,slice,0], vmin=0, vmax=2, cmap='hot')
    plt.subplot(2,2,4)
    max_ = np.argmax(predict,3)
    plt.imshow(max_[:,:,slice], vmin=0, vmax=2, cmap='hot')

index = 1
data = np.load('./sample/X_test_{}.npy'.format(index))   # Only 2 channels
segment = np.load('./sample/y_test_{}.npy'.format(index)) # Only 2 channels
predict = np.load('./sample/mask_output_{}.npy'.format(index)) # Only 2 channels

frequencyTable(segment)
frequencyTable(np.argmax(predict,3))

locateCenter(segment)

showImage(data, segment, predict, slice=55, cmap_='hot', vmin=None, vmax=None)
showImage(data, segment, predict, slice=55, cmap_='hot', vmin=None, vmax=None)




WMHLabel[WMHLabel == 2].shape
EE = np.argmax(predLabel, 3)
EE[EE == 1] # for 2 channels
EE[EE == 2] # for 3 channels

showScanImage2(50, pic=WMHpic, label=WMHLabel, pred=predLabel)

showScanImage2(49, pic=WMHpic, label=WMHLabel, pred=predLabel, 
               saturate=1e-1, threshold=None)