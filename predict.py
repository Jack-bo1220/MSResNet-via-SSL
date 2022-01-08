import argparse
import logging
import os
import time
from xml.dom.minidom import Document
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


import numpy as np
import pydensecrf.densecrf as dcrf
from skimage.io import imread, imsave
from skimage.color import gray2rgb
import time

from utils.dataset import BasicDataset
from network import MSResNet

def predict_img(net,
                full_img,
                device,
                scale_factor=1):
    net.eval()

    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)
        #output_aux,output=net(img)
        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)
        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(full_img.size[1]),
                transforms.ToTensor()
            ]
        )
        #print(full_img.size[1])

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    return full_mask


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='')
    parser.add_argument('--input', '-i', default='',
                        metavar='INPUT',
                        help='file path of input images')
    parser.add_argument('--output', '-o', default='./PredictResult',  
                        metavar='OUTPUT',
                        help='file path of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=1.0)

    return parser.parse_args()


def get_output_filenames(args):
    in_files = args.input
    out_files = []
    for f in in_files:
        pathsplit = os.path.splitext(f)
        out_files.append(pathsplit[0] + '_gt.png')
    return out_files


if __name__ == "__main__":
    t0=time.clock()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  
    rq = time.strftime('%Y%m%d_%H%M', time.localtime(time.time()))
    log_path = os.getcwd() + '/Logs/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_name = log_path + rq + '.log'
    logfile = log_name
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d]: %(message)s")

    fh.setFormatter(formatter)
    logger.addHandler(fh)

    args = get_args()


    in_filePath = args.input
    logging.info("Input file path: {} ".format(in_filePath))
    args.input = os.listdir(in_filePath)
    args.input.sort()
    args.input.sort(key=lambda x: len(x))
    in_files = []
    for i in args.input:
        in_files.append(in_filePath + '/' + i)
    logging.info("Get Input files path successfully!")
    logging.info('Input files list:\n' + str(in_files))
    out_filePath = args.output + '/' + time.strftime('%Y%m%d_%H%M', time.localtime(time.time()))

    if not os.path.exists(out_filePath):
        os.makedirs(out_filePath)

    out_files = get_output_filenames(args)
    logging.info('Output file path' + out_filePath)

    net = MSResNet()
    net.n_classes = 1
    net.n_channels = 3

    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !\n")

    for ii, fn in enumerate(in_files):
        logging.info("\nPredicting image: {} ...".format(fn))

        img = Image.open(fn)
        img0 = Image.open(fn)#
        img_rows,img_cols=img.size
        img = img.resize((448, 448), Image.ANTIALIAS)
        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           device=device)

        mask = mask > args.mask_threshold
        if not args.no_save:
            out_fn = out_files[ii]
            out_fn = out_filePath + '/' + out_fn
            result = Image.fromarray((mask * 255).astype(np.uint8))
            result = result.resize((img_rows, img_cols), Image.ANTIALIAS)

            rows, cols = result.size
            pixel=result.load()
            for i in range(rows):
                for j in range(cols):
                    if  pixel[i,j]<= 127:
                        pixel[i,j]=0
                    else:
                        pixel[i,j] = 255



            result.save(out_fn)

