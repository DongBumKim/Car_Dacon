import argparse


# python train.py --arguments

def argument_parser():
    parser = argparse.ArgumentParser(description="NN Assignment",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)



    #Choose mode for selecting between two models (1 hidden layer, 3hidden layer)
    parser.add_argument("--num_workers", type=int, default=4) # determines how much data will be preprocessed for sending to gpu at once -> more number, faster preprocessing(pallelized) but too much will use too much memory, making the actual training slower
    # Maximum Epochs
    parser.add_argument("--epoch", type=int, default=10)
    
    parser.add_argument("--num_classes", type=int, default=34)
    
    parser.add_argument("--image_size", type=int, default=256)
    
    # Batch Size
    parser.add_argument("--batch_size", type=int, default=4)
    # Choose Optimizer
    parser.add_argument("--optim", type=str, default = "Adam", choices=["SGD","Adam"])
    
    # Choose Model
    parser.add_argument("--model", type=str, default = "faster-rcnn", choices=["faster-rcnn","YOLO"])
    
    # Choose Initial Learning Rate
    
    parser.add_argument("--lr", type=float, default=3e-4)
    

    parser.add_argument("--random_seed",type=int,default = 41)
    

    
    return parser
