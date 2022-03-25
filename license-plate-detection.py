# -*- coding: utf-8 -*-
"""  
License plate detector

Usage:
  license-plate-detection.py <inputDir> <wpodnetPath> [--outDir=<od>] [--lpThresh=<lp>]
  license-plate-detection.py -h | --help

  <inputDir>           Directory with the images to process (*.jpg)
  <wpodnetPath>        Model file

Options:
  --outDir=<od>        Output dir for result files [default: .]
  --lpThresh=<lp>      lp threshold [default: 0.5]
"""

import sys, os
import keras
import cv2
import traceback

from src.keras_utils      import load_model
from glob                 import glob
from os.path              import splitext, basename
from src.utils            import im2single
#from src.keras_utils      import load_model, detect_lp
from src.keras_utils      import load_model, get_lp_oboxes
from src.label            import Shape, writeShapes

from docopt import docopt
import pprint as pp

def adjust_pts(pts,lroi):
    return pts*lroi.wh().reshape((2,1)) + lroi.tl().reshape((2,1))


if __name__ == '__main__':
    # read arguments
    args = docopt(__doc__)
    
    input_dir     = args["<inputDir>"]
    wpod_net_path = args["<wpodnetPath>"]
    output_dir    = args['--outDir']
    lp_threshold  = float(args['--lpThresh'])


    resize_factor = 288.0
    net_step = 2**4
    max_side = 608

    try:
        
        wpod_net = load_model(wpod_net_path)

        imgs_paths = glob('%s/*.jpg' % input_dir)

        print ('Searching for license plates using WPOD-NET')

        for i,img_path in enumerate(imgs_paths):

            print ('\t Processing %s' % img_path)

            bname = splitext(basename(img_path))[0]
            Ivehicle = cv2.imread(img_path)

            #ratio = float(max(Ivehicle.shape[:2]))/min(Ivehicle.shape[:2])
            #side  = int(ratio*288.)
            #bound_dim = min(side + (side%(2**4)),608)
            #print ("\t\tBound dim: %d, ratio: %f" % (bound_dim,ratio))

            #Llp,LlpImgs,_ = detect_lp(wpod_net,im2single(Ivehicle),bound_dim,2**4,(240,80),lp_threshold)
            oboxes = get_lp_oboxes(wpod_net,im2single(Ivehicle),2**4,lp_threshold)
            pp.pprint (oboxes)
            
            
            #for ii in range(len(LlpImgs)):
            #    Ilp = LlpImgs[ii]
            #    Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
            #    Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)
            #
            #    s = Shape(Llp[ii].pts)
            #
            #    cv2.imwrite('%s/%s_%d_lp.png' % (output_dir,bname,ii),Ilp*255.)
            #    writeShapes('%s/%s_%d_lp.txt' % (output_dir,bname,ii),[s])

    except:
        traceback.print_exc()
        sys.exit(1)

    sys.exit(0)


