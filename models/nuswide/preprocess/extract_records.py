import argparse
import numpy as np
import logging
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
    logfile=os.path.splitext(sys.argv[0])[0]+'.pylog'
    fh = logging.FileHandler(logfile)
    fh = logging.FileHandler(logfile)
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
    labelname=open(os.path.join(output_folder, 'label-dict.txt'),'w')
    tmp=np.loadtxt(os.path.join(label_folder,files[0]))
    labelmat=np.zeros((tmp.shape[0],len(files)))
    logger.info('There are %d labels, %d images' %(labelmat.shape))
    for classid, f in enumerate(files):
        labelname.write('%d %s\n' % (classid, f.split('_')[-1].split('.')[0]))
        labelmat[:,classid]=np.loadtxt(os.path.join(label_folder, f), dtype=int)
    #np.save(os.path.join(output_folder,'alllabelmat'), labelmat)
    logger.info("max labels assocaited to one image is %d", np.max(np.sum(labelmat,1)))
    return labelmat

def GetNoneZeroIndex(mat):
    return set(np.where(np.sum(mat,1)>0)[0])

if __name__=="__main__":
    parser=argparse.ArgumentParser(description='extract records from nuswide dataset (ad hoc!); remove images with zero tag in tag1k or zero label')
    parser.add_argument('-l','--labelfolder',  help='merge labels in class files to a matrix, one img per row; input class file folder')
    parser.add_argument('-t','--tag1kfile',  help='tag1k file to filter images with zero tag')
    parser.add_argument('-f','--tagfile',  help='tag file for all images; one line per image')
    parser.add_argument('-i','--imagelist',  help='image list file; Must Check the path!')
    parser.add_argument('-d','--imagefolder', help='image files top folder')
    parser.add_argument('-o','--output', help='folder for all output prefix')
    args=parser.parse_args()
    SetupLogging()
    return

    labelmat=MergeLabelsToMatrix(args.labelfolder, args.output)
    imgidx_nonzero_label=GetNoneZeroIndex(labelmat)
    logger.info('There are %d images with at least one label ' % len(imgidx_nonzero_label))

    imgidx_nonzero_tag=GetNoneZeroIndex(np.loadtxt(args.tag1kfile, dtype=int))
    logger.info('There are %d images with at least one tag ' % len(imgidx_nonzero_tag))

    valididx=list(imgidx_nonzero_label.intersection(imgidx_nonzero_tag))
    logger.info('There are %d images with at least one label and tag ' % len(valididx))
    logger.info('random shuffle index')
    random.shuffle(valididx)

    imglist_fd=open(args.imagelist)
    imglist=[entry.replace('C:\\ImageData\\Flickr\\','').replace('\\','/').strip() for entry in imglist_fd.readlines()]
    tag_fd=open(args.tagfile, 'r')
    taglines=tag_fd.readlines()

    recordfile=os.path.join(args.output,'record-tagwords.dat')
    with open(recordfile, 'w') as fd:
        logger.info('Writing records into %s' %  recordfile)
        logger.info('recordlist.txt:<index in the original image list> <relative path to image> [label id among 81 labels in the alllabel.txt file]#$$#[tags ]\n')
        for idx in valididx:
            tags=taglines[idx]
            imgpath=imglist[idx]
            assert(tags.split()[0]==os.path.splitext(os.path.basename(imgpath))[0].split('_')[1])
            fd.write('%d %s %s#$$#%s\n' %(idx, imgpath,' '.join(['%d' % x for x in np.where(labelmat[idx]>0)[0]]), ' '.join(tags.split()[1:])))
        fd.flush()
        fd.close()

    CheckImageFiles(recordfile, args.imagefolder)
    logger.info("Finished Preprocessing")
