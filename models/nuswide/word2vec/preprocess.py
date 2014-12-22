import argparse
import numpy as np
import logging
import itertools
import os
import sys
import random


logger=logging.getLogger()

def SetupLogging():
    logger.setLevel(logging.DEBUG)
    format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(format)
    logger.addHandler(ch)
    fh = logging.FileHandler(os.path.split(os.path.basename(sys.argv[0]))[0]+'.log')
    fh.setFormatter(format)
    logger.addHandler(fh)

def CheckImageFiles(recordpath, imgfolder):
    count =0
    with open(recordpath) as fd:
        logger.info('Check images from %s' % recordpath)
        lineid=0
        for line in fd:
            imgpath=line.split(':')[0].split(' ')[1]
            if not os.path.exists(os.path.join(imgfolder, imgpath)):
                logger.info('%s does not exist' % imgpath)
                count+=1
            lineid+=1
        logger.info('There are %d images in total, %d missing images' % (lineid, count))

def MergeLabelsToMatrix(label_folder, output_folder):
    files=os.listdir(label_folder)
    files.sort()
    labelname=open(os.path.join(output_folder, 'alllabel.txt'),'w')
    tmp=np.loadtxt(os.path.join(label_folder,files[0]))
    labelmat=np.zeros((tmp.shape[0],len(files)))
    logger.info('There are %d labels, %d images' %(labelmat.shape))
    for classid, f in enumerate(files):
        labelname.write('%d\t%s\n' % (classid, f))
        labelmat[:,classid]=np.loadtxt(os.path.join(label_folder, f), dtype=int)
    #np.save(os.path.join(output_folder,'alllabelmat'), labelmat)
    logger.info("max labels assocaited to one image is %d", np.max(np.sum(labelmat,1)))
    return labelmat

def GetNoneZeroIndex(mat):
    return set(np.where(np.sum(mat,1)>0)[0])

if __name__=="__main__":
    parser=argparse.ArgumentParser(description='prepare nuswide dataset; remove images with zero tag in tag1k or zero label; generate train and test datasets.')
    parser.add_argument('-l','--labelfolder',  help='merge labels in class files to a matrix, one img per row; input class file folder')
    parser.add_argument('-i','--imagelist',  help='image list file')
    parser.add_argument('-t','--tag1kfile',  help='tag1k file')
    parser.add_argument('-f','--tagfile',  help='tag file for all images')
    parser.add_argument('-d','--imagefolder', help='image files top folder')
    parser.add_argument('-o','--output', help='folder for all output prefix')
    parser.add_argument('-n', type=int, help='num of train images to generate, the rest images are for validation and test, half partition')
    args=parser.parse_args()
    SetupLogging()
    labelmat=MergeLabelsToMatrix(args.labelfolder, args.output)
    imgidx_nonzero_label=GetNoneZeroIndex(labelmat)
    logger.info('There are %d images with at least one label ' % len(imgidx_nonzero_label))
    imgidx_nonzero_tag=GetNoneZeroIndex(np.loadtxt(args.tag1kfile, dtype=int))
    logger.info('There are %d images with at least one tag ' % len(imgidx_nonzero_tag))
    imglist_fd=open(args.imagelist)
    imglist=[entry.replace('C:\\ImageData\\Flickr\\','').replace('\\','/').strip() for entry in imglist_fd.readlines()]
    valididx=list(imgidx_nonzero_label.intersection(imgidx_nonzero_tag))
    logger.info('There are %d images with at least one label and tag ' % len(valididx))
    with open(os.path.join(args.output,'recordlist.txt'), 'w') as fd:
        logger.info('recordlist.txt:index in the original image list; relative path to image; label id among 81 labels in the alllabel.txt file\n')
        for idx in valididx:
            fd.write('%d %s: %s\n' %(idx, imglist[idx],' '.join(['%d' % x for x in np.where(labelmat[idx]>0)[0]])))
        fd.flush()
        fd.close()

    train_idx=[valididx[idx] for idx in list(set(np.random.randint(len(valididx),size=2*args.n)))[0:args.n]]
    test_idx=list(set(valididx)-set(train_idx))
    logger.info('Train set size %d, Test set size %d' %(len(train_idx), len(test_idx)))
    tag_fd=open(args.tagfile, 'r')
    taglines=tag_fd.readlines()

    trainrecordfile=os.path.join(args.output,'train.dat')
    with open(trainrecordfile, 'w') as fd:
        logger.info('Writing records into %s' %  trainrecordfile)
        random.shuffle(train_idx)
        for idx in train_idx:
            tags=taglines[idx]
            imgpath=imglist[idx]
            assert(tags.split()[0]==os.path.splitext(os.path.basename(imgpath))[0].split('_')[1])
            fd.write('%d %s %s#$$#%s\n' %(idx, imgpath,' '.join(['%d' % x for x in np.where(labelmat[idx]>0)[0]]), ' '.join(tags.split()[1:])))
        fd.flush()
        fd.close()

    testrecordfile=os.path.join(args.output,'test.dat')
    with open(testrecordfile, 'w') as fd:
        logger.info('Writing records into %s' %  testrecordfile)
        random.shuffle(test_idx)
        for idx in test_idx:
            tags=taglines[idx]
            imgpath=imglist[idx]
            assert(tags.split()[0]==os.path.splitext(os.path.basename(imgpath))[0].split('_')[1])
            fd.write('%d %s: %s#$$#%s\n' %(idx, imgpath,' '.join(['%d' % x for x in np.where(labelmat[idx]>0)[0]]), ' '.join(tags.split()[1:])))
        fd.flush()
        fd.close()

    CheckImageFiles(trainrecordfile, args.imagefolder)
    CheckImageFiles(testrecordfile, args.imagefolder)
    logger.info("Finished Preprocessing; Train and test data are in %s and %s" %(os.path.join(args.output,'train.dat'),os.path.join(args.output,'test.dat')))
