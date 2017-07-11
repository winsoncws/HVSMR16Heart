# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 12:24:48 2017

@author: winsoncws
"""
import sys
import tensorgraph as tg
import tensorflow as tf
from tensorgraph.cost import entropy, accuracy, iou, smooth_iou
from dataHVSMR16 import HVSMRdataset
import model # all model
from scipy.misc import imsave
import numpy as np
#from scipy.ndimage.interpolation import rotate

if __name__ == '__main__':
    
   
    #print(sys.argv[0]) # input from terminal
    #print(sys.argv[1]) # input from terminal
    #print(sys.argv[2]) # input from terminal
    
    learning_rate = 0.001
    
    max_epoch = 100
    es = tg.EarlyStopper(max_epoch=max_epoch,
                         epoch_look_back=3,
                         percent_decrease=0)


    dataset = HVSMRdataset('./datasetHVSMR16Heart')
    assert dataset.AbleToRetrieveData(), 'not able to locate the directory of dataset'
    dataset.InitDataset(splitRatio=1.0, shuffle=True)         # Take everything 100%
#    X_ph = tf.placeholder('float32', [None, 84, 256, 256, 1])  #float32
#    y_ph = tf.placeholder('uint8', [None, 84, 256, 256, 1])
    X_ph = tf.placeholder('float32', [None, None, None, None, 1], name='XX')  #float32
    y_ph = tf.placeholder('uint8', [None, None, None, None, 1], name='YY')
    
    y_ph_cat = tf.one_hot(y_ph,3) # --> unstack into 3 categorical Tensor [?, 84, 256, 256, 1, 3]
    y_ph_cat = y_ph_cat[:,:,:,:,0,:]
    #y_ph_cat = tf.reduce_max(y_ph_cat, 4)   # --> collapse the extra 4th redundant dimension
    #seq = model.model3D()  
    phase = tf.placeholder(tf.bool, name='phase') # Boolen placeholder
    
    print('TRAINING')
    # for one hot
    y_train_sb = model.model(X_ph, train=phase)
    
    #y_train_sb = (seq.train_fprop(X_ph))  
    #y_train_sb = (seq.train_fprop())[0][0]
    print('TESTING')
    y_test_sb = y_train_sb
    #y_test_sb = (seq.test_fprop(X_ph))
    #y_test_sb = (seq.test_fprop())[0][0]
    
    print('TRAINED')

    print(y_ph_cat)
    print(y_test_sb)
    ### CHANGE TO 2 CHANNELS
    train_cost_label =  (1 - smooth_iou(y_ph_cat[:,:,:,:,1] , y_train_sb[:,:,:,:,0]) ) 
    train_cost_others = (1 - smooth_iou(y_ph_cat[:,:,:,:,2] , y_train_sb[:,:,:,:,1]) ) 
    train_cost_sb = tf.reduce_sum([train_cost_label * 0.5,train_cost_others * 0.5])
    #train_cost_sb = train_cost_label
    valid_cost_background = (1 - smooth_iou(y_ph_cat[:,:,:,:,0] , y_test_sb[:,:,:,:,0]) ) # can ignore 
    valid_cost_label = (1 - smooth_iou(y_ph_cat[:,:,:,:,1] , y_test_sb[:,:,:,:,0]) ) 
    valid_cost_others = (1 - smooth_iou(y_ph_cat[:,:,:,:,2] , y_test_sb[:,:,:,:,1]) )
    test_cost_sb = tf.reduce_sum([valid_cost_label * 0.5,valid_cost_others * 0.5])  
    
    
    # ACCURACY
    # CHANGE TO 2 CHANNELS    
    test_accu_sb = iou(y_ph_cat[:,:,:,:,1:], y_test_sb, threshold=0.5)         # Works for Softmax filter2
    
    print('DONE')    

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(train_cost_sb)
    
    # model Saver
    #saver = tf.train.Saver()
    #X_ph = tf.placeholder('float32', [None, None, None, None, 1])  #float32

    
    #gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    #with tf.Session(config = tf.ConfigProto(gpu_options = gpu_options)) as sess:
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        print("INITIALIZE SESSION")
        
        
        #dataset.InitDataset()  # Take everything 80% Train 20% Validation
        
        batchsize = 1  # size=3
        #######
        # Just to train 0 & 1, ignore 2=Other Pathology. Assign 2-->0
        # dataY[dataY ==2] = 0
        #######
        X_train, y_train = dataset.NextBatch3D(10,dataset='train')
        X_test, y_test = dataset.NextBatch3D(10,dataset='validation')
             
        
#        iter_train = tg.SequentialIterator(X_train, y_train, batchsize=batchsize)
#        iter_test = tg.SequentialIterator(X_test, y_test, batchsize=batchsize)
        
        best_valid_accu = 0
        for epoch in range(max_epoch):
            print('epoch:', epoch)
            pbar = tg.ProgressBar(10)
            ttl_train_cost = 0
            ttl_examples = 0
            print('..training')
            
            for i in range(10):
                X_tr = X_train[i]
                y_tr = y_train[i]
                y_tr = np.array(y_tr, dtype='int8')
                
                X_tr = X_tr.reshape((1,)+X_tr.shape+(1,))
                y_tr = y_tr.reshape((1,)+y_tr.shape+(1,))
                feed_dict = {X_ph:X_tr, y_ph:y_tr, phase:1}
                _, train_cost = sess.run([optimizer,train_cost_sb] , feed_dict=feed_dict)              
                ttl_train_cost += train_cost
                ttl_examples += 1
                pbar.update(ttl_examples)
            mean_train_cost = ttl_train_cost/float(10)
            print('\ntrain cost', mean_train_cost)

            ttl_valid_cost = 0
            ttl_valid_accu = 0
            tt_valid_0 = 0
            tt_valid_1 = 0
            tt_valid_2 = 0
            ttl_examples = 0
            pbar = tg.ProgressBar(10)
            print('..validating')
            for i in range(10):
                X_tr = X_train[i]
                y_tr = y_train[i]
                X_tr = X_tr.reshape((1,)+X_tr.shape+(1,))
                y_tr = y_tr.reshape((1,)+y_tr.shape+(1,))
                feed_dict = {X_ph:X_tr, y_ph:y_tr, phase:0}
                valid_cost, valid_accu, valid_0, valid_1, valid_2 = sess.run([test_cost_sb, test_accu_sb, valid_cost_background,
                                                                     valid_cost_label, valid_cost_others],
                                                                     feed_dict=feed_dict)
                #mask_output = sess.run(y_test_sb, feed_dict=feed_dict)
                ttl_valid_cost += valid_cost
                ttl_valid_accu += valid_accu
                tt_valid_0 += valid_0
                tt_valid_1 += valid_1
                tt_valid_2 += valid_2
                ttl_examples += 1
                pbar.update(ttl_examples)
            mean_valid_cost = ttl_valid_cost/float(ttl_examples)
            mean_valid_accu = ttl_valid_accu/float(ttl_examples)
            mean_valid_0 = tt_valid_0/float(ttl_examples)
            mean_valid_1 = tt_valid_1/float(ttl_examples)
            mean_valid_2 = tt_valid_2/float(ttl_examples)
            print('\nvalid average cost', mean_valid_cost)
            #print('valid Background', mean_valid_0)
            print('valid Label1', mean_valid_1)
            print('valid Label2', mean_valid_2)
            print('valid accu', mean_valid_accu)
            
            
            if best_valid_accu < mean_valid_accu:
                best_valid_accu = mean_valid_accu

            if es.continue_learning(valid_error=mean_valid_cost, epoch=epoch):
                print('epoch', epoch)
                print('best epoch last update:', es.best_epoch_last_update)
                print('best valid last update:', es.best_valid_last_update)
                print('best valid accuracy:', best_valid_accu)
            else:
                print('training done!')
                break
        
        #save_path = saver.save(sess, "./trainModel/model1/trained_model.ckpt")    
        #print("Model saved in file: %s" % save_path)
        
        
        ### 1ST PREDICTION
        #predictIndex = sys.argv[1] # input from terminal
        for i in range(4):
            print('Prediction 3D Scan of No #'+str(i))        
            intIndex = int(i)  
            X_tr = X_train[i]
            y_tr = y_train[i]
            X_tr = X_tr.reshape((1,)+X_tr.shape+(1,))
            y_tr = y_tr.reshape((1,)+y_tr.shape+(1,))
            feed_dict = {X_ph:X_tr, y_ph:y_tr, phase:0}        
            
            mask_output = sess.run(y_test_sb, feed_dict=feed_dict)
    
            print('mask_outpt type')        
            print(type(mask_output))
            print(mask_output.shape)        
            
            np.save('X_test_'+str(i)+'.npy',X_train[i])
            np.save('y_test_'+str(i)+'.npy',y_train[i])
            np.save('mask_output_'+str(i)+'.npy',mask_output[0])
        
        
        