import numpy as np
from matplotlib import pyplot as plt
import argparse

def Sample(gnd, c):
  n,m=gnd.shape
  cidx=np.arange(m)
  np.random.shuffle(cidx)
  scidx=cidx[0:c]
  scidx=[16,5,17,7]
  print scidx
  ret=[]
  label=[]
  for gid, g in enumerate(gnd):
    if np.sum(g[scidx])==1:
      ret.append(gid)
      label.append(np.where(g[scidx]>0)[0][0])
  #print 'num of samples is %d, num of label %d' % (len(ret), len(label))
  #hist,_=np.histogram(label, bins=4)
  #print hist
  return ret, label

def Plot(origin, mapped, coloridx):
  print origin.shape, len(coloridx)
  stdcolors=['b','g','#CCFF33','w']
  #['#000066','b','#CCFF33','#003300','#FFCCFF','#800000']
  #np.random.rand(len(labels))

  colors=[stdcolors[color] for color in coloridx]
  plt.subplot(2,1,1)
  plt.scatter(origin[:, 0], origin[:,1], s=65.0, c=colors, alpha=0.3)
  plt.subplot(2,1,2)
  plt.scatter(mapped[:, 0], mapped[:,1], s=65.0, c=colors, alpha=0.3)
  fig=plt.gcf()
  dftsize=fig.get_size_inches()
  print 'dpi %i, default size in inches: %i %i' % (fig.get_dpi(), dftsize[0], dftsize[1])
  fig.set_size_inches(dftsize[0]*1, dftsize[1]*1)

  plt.show()

if __name__=="__main__":
  parser=argparse.ArgumentParser(description='Plot images on 2-d space, different color for different labels')
  parser.add_argument('data',nargs=2, help='image data files, plain text, each line has two float value')
  parser.add_argument('label', help="image label file, npy file, each line is the tag occurrence of one image")
  parser.add_argument('--m', type=int, default=4, help='num of labels to plot')
  args=parser.parse_args()

  labels=np.load(args.label)
  Loop=True
  while Loop:
    idx, coloridx=Sample(labels, args.m)
    hist,_=np.histogram(coloridx, bins=4)
    print hist
    if np.std(hist)<500:
      Loop=False
      origin=np.loadtxt(args.data[0])
      mapped=np.loadtxt(args.data[1])
      Plot(origin[idx], mapped[idx], coloridx)
