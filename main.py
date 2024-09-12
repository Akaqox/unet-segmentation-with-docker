import sys
sys.dont_write_bytecode = True
import ssl 
ssl._create_default_https_context = ssl._create_unverified_context
import argparse

from datetime import datetime
import gc
import torch
torch.cuda.empty_cache()
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from utils.loss import BinaryDiceLoss, MergeLosses, IoULoss, FocalTverskyLoss
from dataset.dataset import LeafDataset
from train.train import Train
from model.model import UNet, UNet34, UNet50
# from utils.utils import tensor2image
                                                                                                                                                                                                                                                                                                                                                                                     
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
# writer = SummaryWriter('results/reports/catdog_trainer_{}'.format(timestamp))

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',default='unet50', help='Select the model architecture.(unet, unet34, unet50) Default: unet50')
    parser.add_argument('--loss',default='dice', nargs='+', help='Select the one or two loss.(dice, bce, tversky, iou) Default: Dice')
    parser.add_argument('--lw', type=float, default=0.5, help='Select the weight of first loss of  multiple loss. Default: 0.5')
    parser.add_argument('--alpha', type=float, default=0.7, help='Select the tversky loss if selected alpha value. Default: 0.7')
    parser.add_argument('--opt', default='adam', help='For now single option is adam. Default:adam')
    parser.add_argument('--lr', type=float, default=0.00005, help='Select the learning rate. Default:0.00005')
    parser.add_argument('--bs', type=int, default=8, help='Select the batch size. Default:8')
    parser.add_argument('--epoch', type=int, default=15, help='Select the epoch number. Default:15')
    parser.add_argument('--worker', type=int, default=0, help='Select the number of workers. Default:0')
    
    args = parser.parse_args()
    return args


def hyper_parameters(args):

    print(f"Selected Model Architecture: {args.model}")
    print(f"Selected Loss Function(s): {args.loss}")
    print(f"Weight of First Loss: {args.lw}")
    print(f"Alpha for Tversky Loss: {args.alpha}")
    print(f"Selected Optimizer: {args.opt}")
    print(f"Learning Rate: {args.lr}")
    print(f"Batch Size: {args.bs}")
    print(f"Epochs: {args.epoch}")
    print(f"Number of Workers: {args.worker}")

    if args.model == 'unet':
        model = UNet(n_classes=1)
    elif args.model == 'unet34':
        model = UNet34(n_classes=1)
    elif args.model == 'unet50':
        model = UNet50(n_classes=1)
    else:
        assert not model == None, 'Model input is not recognized'

    loss_dict = {
        'dice' : BinaryDiceLoss(), 
        'bce' : torch.nn.BCELoss(), 
        'iou' : IoULoss(), 
        'tversky' : FocalTverskyLoss(alpha = args.alpha)}

    if type(args.loss) == str:
        criterion = loss_dict[args.loss]

    elif len(args.loss) == 2:
        loss1 = loss_dict[args.loss[0]]
        loss2 = loss_dict[args.loss[1]]
        criterion = MergeLosses(loss1, loss2, args.lw)
    else:
        assert not model == None, 'Loss input is not recognized'

    if args.opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        assert not model == None, 'Optimizer input is not recognized'
    #Create dataset classes


    hyper_p =[
            args.lr, 
            args.bs, 
            0.1, 
            model.__class__.__name__, 
            optimizer.__class__.__name__, 
            criterion.__class__.__name__
            ]
    return args.bs, args.worker, hyper_p, model, criterion, optimizer, args.epoch

if __name__ == "__main__":    
    args = argparser()
    dataset_path = 'dataset/'

    train = LeafDataset(image_dir=dataset_path, sub='train')
    val = LeafDataset(image_dir=dataset_path, sub='valid')

    batch_size, num_workers, hyper_p, model, criterion, optimizer, EPOCH = hyper_parameters(args)

    #Create Dataloaders
    trainLoader = DataLoader(train, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    valLoader = DataLoader(val, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    #memory allocation
    del train
    del val
    gc.collect

    train_model = Train(trainLoader, valLoader, hyper_p, model=model, criterion=criterion, optimizer=optimizer, EPOCH=EPOCH)
    train_model.fit()
    train_model.plot()
    


