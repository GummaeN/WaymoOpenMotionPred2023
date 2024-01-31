import numpy as np
import cv2
import pandas as pd
from google.protobuf import text_format
from google.colab.patches import cv2_imshow


import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap


def visualize(raster, pred, cluster_pred, target):


  raster = np.transpose(np.squeeze(raster),(1,2,0))
  target = np.array(np.squeeze(target))
  cluster_pred = np.array(np.squeeze(cluster_pred))
  img = np.zeros((224, 224, 3), dtype=np.uint16)
  img_over = np.zeros((224, 224, 3), dtype=np.uint16)


  img[:,:,0] = raster[:,:,2]*1500
  img[:,:,1] = raster[:,:,2]*1500
  img[:,:,2] = raster[:,:,2]*1500

  img[:,:,0][img[:,:,0] > 255] = 170
  img[:,:,1][img[:,:,1] > 255] = 170
  img[:,:,2][img[:,:,2] > 255] = 170


  img_over[:,:,1] = raster[:,:,7]*255+raster[:,:,12]*200
  img_over[:,:,2] = (raster[:,:,1]+raster[:,:,0])*255
  img_over[:,:,2][img_over[:,:,2] > 255] = 255
  img_over[:,:,1][img_over[:,:,1] > 255] = 255

  img = cv2.addWeighted(img_over, 1, img, 1, 0)

  
  plt.ioff()
  px = 1/plt.rcParams['figure.dpi']  # pixel in inches
  fig, ax = plt.subplots(figsize=(450*px, 450*px))
  plt.style.use('dark_background')
  
  def colorlist2(c1, c2, num):
      l = np.linspace(0,1,num)
      a = np.abs(np.array(c1)-np.array(c2))
      m = np.min([c1,c2], axis=0)
      s  = np.sign(np.array(c2)-np.array(c1)).astype(int)
      s[s==0] =1
      r = np.sqrt(np.c_[(l*a[0]+m[0])[::s[0]],(l*a[1]+m[1])[::s[1]],(l*a[2]+m[2])[::s[2]]])
      return r

  for i in range(len(cluster_pred)):
    x = cluster_pred[i,:,0]
    y = cluster_pred[i,:,1]

    t1 = 	(206,162,253)
    t2 = 	(83,0,100)


    c1 = tuple(i /255 for i in t1)
    c2 = tuple(i /255 for i in t2)

    cmap = LinearSegmentedColormap.from_list("", colorlist2(c1, 	c2,100))

    points = np.array([x, y]).T.reshape(-1,1,2)
    segments = np.concatenate([points[:-2],points[1:-1], points[2:]], axis=1)

    lc = LineCollection(segments, cmap=cmap, linewidth=4, alpha = 0.65)
    lc.set_array(x)
    line = ax.add_collection(lc)
  

  x = target[:,0]
  y = target[:,1]
  plt.plot(x,y,color = 'yellow',linewidth=2, zorder = 2)


  coordList = (29,49,79)
  th_lat = [2,3.6,6.0]
  th_long = [4.0,7.2,12]
  k = 0
  r = []
  for i in coordList:
    
    x = target[i,0]
    y = target[i,1]

    hyp = (np.sqrt(((th_lat[k]/2)**2)+((th_long[k]/2)**2)))
    ang = np.arctan((x-target[i-1,0])/(y-target[i-1,1]))
  
    ang2 = -(np.pi-np.arcsin((th_lat[k]/2)/hyp))
    ang3 = ang2 + ang
    x0 = x + hyp*np.sin(ang3)
    y0 = y + hyp*np.cos(ang3)

    r.append(patches.Rectangle((x0,y0),th_lat[k] ,th_long[k],color="yellow", alpha=0.50, fill = True, zorder = 4))
    t1 = mpl.transforms.Affine2D().rotate_around(x0, y0,-ang)
    r[k].set_transform(t1 + ax.transData)

    ax.add_patch(r[k])

    k += 1

  plt.xlim([0, 224])
  plt.ylim([0, 224]) 
  plt.setp([ax.get_xticklines() + ax.get_yticklines() + ax.get_xgridlines() + ax.get_ygridlines()],antialiased=False)
  plt.rcParams['text.antialiased']=False
  ax.axis('off')
  fig.tight_layout(pad=0)

  ax.margins(0)
  fig.canvas.draw()



  data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
  data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  data = data.astype(np.uint16)
  tmp_data = data.copy()
  data[:,:,0] = tmp_data[:,:,2]
  data[:,:,2] = tmp_data[:,:,0]

  res = cv2.resize(img, (450,450), interpolation=cv2.INTER_LINEAR)

  data[:,:,0][data[:,:,0] > 255] = 230
  data[:,:,1][data[:,:,1] > 255] = 255
  data[:,:,2][data[:,:,2] > 255] = 255


  img_new = cv2.addWeighted(data, 1, res, 1, 0)

  img_new[img_new == 0] = 15



  cv2_imshow(img_new)