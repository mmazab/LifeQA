#!/usr/bin/python
import os
import sys
import json
import datetime
import numpy as np
import time
from collections import Counter
from optparse import OptionParser
from IPython import embed
import config
from tqdm import tqdm


def get_scores_corresponding_objects(input_dir, rate, outdir):
    ext = '.npy'
    clips = os.listdir(input_dir)
    for clip in clips:
        features_path = os.path.join ( input_dir, clip)

        clip_objects = []
        print (' Currently processing: {0}'.format(clip))
        for f in tqdm( sorted( os.listdir ( features_path ) ) ) :
            if not f.endswith (ext) : continue
            feature_file = os.path.join ( features_path, f)
            
            features = np.load( feature_file, allow_pickle=True).item()
            if not features:
                print (" Empty feature file encountered: {0} ".format( feature_file) )
                continue

            scores = features['cls_scores']
            cls_indexes = np.argmax(scores, axis=1) 
             
            if not os.path.isdir ( outdir ):
                os.mkdir( outdir )

            #if not os.path.isdir( os.path.join( outdir, clip ) ):
            #    os.mkdir(  os.path.join(outdir, clip)   )

            with open (  config.objects_vocab, 'r' ) as fi:
                lines = fi.readlines()
                index_to_object = { i: l.strip() for i, l in enumerate(lines)  }
                
            frame_objects = set( [ index_to_object[index] for index in cls_indexes if index>0 ] )
            clip_objects.append ( [ f[:-4], list( frame_objects ) ]  )

        with open( os.path.join ( outdir, clip+'.json'), 'w')  as out_f:
            json.dump ( clip_objects, out_f, indent=4, sort_keys=True ) 


def init_option_parser():
    """Initialize parser."""
    usage = """ Check out the code for options. """
    parser = OptionParser(usage=usage)
    parser.add_option("-i", "--input_features_dir", action="store", type="string", default="", help="path to the video features")
    parser.add_option("-n", "--rate", action="store", type=int, default=10, help="Get a frame every from every n frames")
    parser.add_option("-o", "--outdir", action="store", type="string", default="output", help="output directory")
    return parser


if __name__ == '__main__':

    ### Parse command line options
    parser = init_option_parser()
    opts, args = parser.parse_args(sys.argv)
    print ( opts.input_features_dir )
    get_scores_corresponding_objects( opts.input_features_dir, opts.rate, opts.outdir )
