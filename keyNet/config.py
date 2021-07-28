import argparse

def get_config():
    parser = argparse.ArgumentParser(description='Train Key.Net Architecture')
    
    ## basic configuration
    parser.add_argument('--data-dir', type=str, default='/home/jongmin/datasets/ImageNet2012/ILSVRC2012_img_val', #default='path-to-ImageNet',
                            help='The root path to the data from which the synthetic dataset will be created.')
    parser.add_argument('--synth-dir', type=str, default='datasets/synth/',
                            help='The path to save the generated sythetic image pairs.')
    parser.add_argument('--weights-dir', type=str, default='keyNet/weights',
                            help='The path to save the Key.Net weights.')
    parser.add_argument('--write-summary', type=bool, default=False,
                            help='Set to True if you desire to save the summary of the training.')
    parser.add_argument('--network-version', type=str, default='KeyNet_default',
                            help='The Key.Net network version name')
    parser.add_argument('--random-seed', type=int, default=12345,
                            help='The random seed value for PyTorch and Numpy.')
    parser.add_argument('--num-epochs', type=int, default=30,
                        help='Number of epochs for training.')
    parser.add_argument('--epochs-val', type=int, default=3,
                        help='Set the number of training epochs between repeteability checks on the validation set.')
    ## Dataset generation
    parser.add_argument('--patch-size', type=int, default=192,
                        help='The patch size of the generated dataset.')
    parser.add_argument('--max-angle', type=int, default=45,
                        help='The max angle value for generating a synthetic view to train Key.Net.')
    parser.add_argument('--max-scale', type=int, default=2.0,
                        help='The max scale value for generating a synthetic view to train Key.Net.')
    parser.add_argument('--max-shearing', type=int, default=0.8,
                        help='The max shearing value for generating a synthetic view to train Key.Net.')
    parser.add_argument('--is-debugging', type=bool, default=False,
                        help='Set variable to True if you desire to train network on a smaller dataset.')
    parser.add_argument('--load-tfrecord', type=bool, default=False,
                        help='Load tensorflor tfrecord.')                        
    ## Training 
    parser.add_argument('--batch-size', type=int, default=32,
                        help='The batch size for training.')
    parser.add_argument('--init-initial-learning-rate', type=float, default=1e-3,
                        help='The init initial learning rate value.')
    parser.add_argument('--num-epochs-before-decay', type=int, default=20,
                        help='The number of epochs before decay.')
    parser.add_argument('--learning-rate-decay-factor', type=float, default=0.5,
                        help='The learning rate decay factor.')
    parser.add_argument('--resume-training', type=str, default='',
                        help='Set saved model parameters if resume training is desired.')
    parser.add_argument('--weight-coordinates', type=bool, default=True,
                        help='Weighting coordinates by their scores.')  
    parser.add_argument('--MSIP_sizes', type=str, default="8,16,24,32,40",
                        help='MSIP sizes.')
    parser.add_argument('--MSIP_factor_loss', type=str, default="256.0,64.0,16.0,4.0,1.0",
                        help='MSIP loss balancing parameters.')
    ## network architectures
    parser.add_argument('--num-filters', type=int, default=8,
                        help='The number of filters in each learnable block.')
    parser.add_argument('--num-learnable-blocks', type=int, default=3,
                        help='The number of learnable blocks after handcrafted block.')
    parser.add_argument('--num-levels-within-net', type=int, default=3,
                        help='The number of pyramid levels inside the architecture.')
    parser.add_argument('--factor-scaling-pyramid', type=float, default=1.2,
                        help='The scale factor between the multi-scale pyramid levels in the architecture.')
    parser.add_argument('--conv-kernel-size', type=int, default=5,
                        help='The size of the convolutional filters in each of the learnable blocks.')
    parser.add_argument('--nms-size', type=int, default=15,
                        help='The NMS size for computing the validation repeatability.')
    parser.add_argument('--border-size', type=int, default=15,
                        help='The number of pixels to remove from the borders to compute the repeatability.')

    args = parser.parse_args()
   
    return args