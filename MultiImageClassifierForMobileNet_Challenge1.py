from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
from glob import glob
#import winsound

import numpy as np
import pandas as pd
#from sklearn.metrics import confusion_matrix
#import matplotlib.pyplot as plt

#import plot_cm 
import tensorflow as tf
#from importlib import reload

def read_tensor_from_image_file(file_name, input_height=299, input_width=299,
				input_mean=0, input_std=255):
  input_name = "file_reader"
  #output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(file_reader, channels = 3,
                                       name='png_reader')
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                  name='gif_reader'))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
  else:
    image_reader = tf.image.decode_jpeg(file_reader, channels = 3,
                                        name='jpeg_reader')
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0);
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  #sess = tf.Session(config=tf.ConfigProto(log_device_placement=True),)
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  config.gpu_options.per_process_gpu_memory_fraction = 0.8
  sess = tf.Session(config=config)
  result = sess.run(normalized)

  return result

#def convert(string):
#   n=len(string)
#   if n==6:
#      x=0 #photos
#   elif n==7:
#      x=1 #invoice
#   elif n==11:
#      x=2 #
#   elif n==12:
#      x=3 #motorlicense
#   elif n==14:
#     if string[0]=='a':
#       x=4 #accidentreport
#     else:
#       x=5 #drivinglicense
#   return x

if __name__ == '__main__':
    
  #Default Input Image settings
  input_height = 224
  input_width = 224
  input_mean = 128
  input_std = 128
    
  #Default layer nomenclature
  #input_layer = "Mul"
  input_layer = "input"
  output_layer = "final_result"
  input_name = "import/" + input_layer
  output_name = "import/" + output_layer
  
  #Define Model File
  dir = os.getcwd()
  #model_file = os.path.join(dir,"model_files","retrained_graph_run_sigmoid.pb")
  model_file = os.path.join(dir, "retrained_graph.pb")
  print("Model file is set as:",os.path.basename(model_file))
  
  #Load Model file
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  input_operation = graph.get_operation_by_name(input_name);
  output_operation = graph.get_operation_by_name(output_name); 
  
  #Define label file
  #dir = os.getcwd()
  #label_file = os.path.join(dir,"label_files","retrained_labels_run_sigmoid.txt")
  label_file = os.path.join(dir, "retrained_labels.txt")
  print("Label file is set as:",os.path.basename(label_file))
  
  #Load label file
  labels = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    labels.append(l.rstrip())
    
  #Classify Multiple Images
  #dir = os.getcwd()
  #mypath = os.path.join(dir,"dataset","original_set","drivinglicense")
  mypath = os.path.join(dir, "test_set")
  print("image directory set as:",os.path.basename(mypath))
  true=glob((mypath+"/*"))
  newDF =  pd.DataFrame(columns=['Original Label','Filename','Primary Result','Primary Score',
                                 'Secondary Result','Secondary Score','Tertiary Result','Tertiary Score'])
  #ytrain=[]
  #ytest=[]
  
  for name in true:
    lb= os.path.basename(name)    
    print('Looking Images in',lb)
    files = os.listdir(name)
    i = 0
    for filename in files:
        file_name = os.path.join(name,filename)
        file_name = str(file_name)
        i = i+1
        if i%10 == 0 :
            print("Processed",i,"images")
        try:
            t = read_tensor_from_image_file(file_name,
                                  input_height=input_height,
                                  input_width=input_width,
                                  input_mean=input_mean,
                                  input_std=input_std)
                   
            with tf.Session(graph=graph) as sess:
                results = sess.run(output_operation.outputs[0],
                                   {input_operation.outputs[0]: t})
            results = np.squeeze(results)
            
            top_k = results.argsort()[-5:][::-1]
            
            primary_score = results[top_k[0]]
            secondary_score = results[top_k[1]]
            tertiary_score = results[top_k[2]]
            
            primary_label = labels[top_k[0]]
            secondary_label = labels[top_k[1]]
            tertiary_label = labels[top_k[2]]
            
            #score = results[0]

            #if np.max(results) < 0.5:
            #        out = "unknown"
            #else:
            #        out = labels[np.argmax(results)]
            
            newDF = newDF.append({'Original Label':lb, 'Filename':os.path.basename(file_name), 
                                  'Primary Result':primary_label,'Primary Score':primary_score,
                                 'Secondary Result':secondary_label,'Secondary Score':secondary_score,
                                 'Tertiary Result':tertiary_label,'Tertiary Score':tertiary_score},
                                    ignore_index=True)
            
            #ytrain+=[convert(str(primary_label))]
            #ytest+=[convert(lb)]
            
        except:
            print('There was some problem in Image data file',os.path.basename(file_name))
            continue  

  output_file = 'ClassificationResult.xlsx'
  print("Output file name given as:",output_file)
  writer = pd.ExcelWriter(output_file)
  newDF.to_excel(writer,'Sheet1')
  writer.save() 
  #con_mat = confusion_matrix(ytest, ytrain)
  #print(con_mat)
  #plt.figure()
  #ticks=['photos','invoice','declaration','motorlicense','accidentreport','drivinglicense']
  #plot_cm.plot_confusion_matrix(con_mat, classes=ticks,
  #                    title='Confusion matrix, without normalization')
  #plt.show()
  #winsound.Beep(1000,5000)
  
 
