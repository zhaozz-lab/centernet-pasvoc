from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorboardX import SummaryWriter
import os
from dataload import listDataset
import torch
import torch.utils.data
from opts import opts
import logging
import time
import torchvision
from torchvision import datasets, transforms

from losses import CtdetLoss
from collections import OrderedDict

logging.basicConfig(
    format='[%(asctime)s %(name)s %(levelname)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO)
# debug<info<warn<Error<Fatal
logger = logging.getLogger(__name__)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num):
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count


def train(epoch, model, optimizer, criterion, train_loader, config, writer):
    global global_step
    global_step = 0
    logger.info('Train {}'.format(epoch))
    model.train()
    loss_meter = AverageMeter()
    start = time.time()
    for step, (image,label) in enumerate(train_loader):
        global_step += 1
        if config['tensorboard_images'] and epoch == 0 and step == 0:
            images = torchvision.utils.make_grid(
                image, normalize=True, scale_each=True)
            writer.add_image('Test/Image', images, epoch)
        
        image = image.cuda()
        model = model.cuda()
        
        optimizer.zero_grad()
        outputs = model(image)

        label['hm'] = label['hm'].cuda()
        label['wh'] = label['wh'].cuda()
        label['ind'] = label['ind'].cuda()
        label['reg_mask'] = label['reg_mask'].cuda()
        label['reg'] = label['reg'].cuda()
        label['hm'] = label['hm'].cuda()

       
        loss,loss_states = criterion(outputs, label)
        loss.mean()
        loss.backward()
        optimizer.step()
        num = image.size(0)
        loss_meter.update(loss.item(), num)
        

        if config['tensorboard']:
            writer.add_scalar('Train/RunningLoss', loss_meter.val, global_step)

        if step % 100 == 0:
            logger.info('Epoch {} Step {}/{} '
                        'Loss {:.4f} ({:.4f}) '.format(
                            epoch,
                            step,
                            len(train_loader),
                            loss_meter.val,
                            loss_meter.avg,
                        ))

    elapsed = time.time() - start
    logger.info('Elapsed {:.2f}'.format(elapsed))

    if config['tensorboard']:
        writer.add_scalar('Train/Loss', loss_meter.avg, epoch)
        writer.add_scalar('Train/Time', elapsed, epoch)


def test(epoch, model, criterion, val_loader, config, writer):
    
    logger.info('Test {}'.format(epoch))
    model.eval()

    loss_meter = AverageMeter()
    angle_error_meter = AverageMeter()
    start = time.time()
    for step, (image,label) in enumerate(val_loader):
        if config['tensorboard_images'] and epoch == 0 and step == 0:
            images = torchvision.utils.make_grid(
                image, normalize=True, scale_each=True)
            writer.add_image('Test/Image', images, epoch)
        
        image = image.cuda()
        model = model.cuda()
        with torch.no_grad():
            output = model(image)


        label['hm'] = label['hm'].cuda()
        label['wh'] = label['wh'].cuda()
        label['ind'] = label['ind'].cuda()
        label['reg_mask'] = label['reg_mask'].cuda()
        label['reg'] = label['reg'].cuda()
        label['hm'] = label['hm'].cuda()

    
        loss = criterion(output, label)
        
        num = image.size(0)
        loss_meter.update(loss[0].item(), num)
        
    elapsed = time.time() - start
    logger.info('Elapsed {:.2f}'.format(elapsed))
    logger.info('Epoch {} Step {}/{} '
                        'Loss {:.4f} ({:.4f}) '.format(
                            epoch,
                            step,
                            len(val_loader),
                            loss_meter.val,
                            loss_meter.avg,
                        ))
    if config['tensorboard']:
        if epoch > 0:
            writer.add_scalar('Test/Loss', loss_meter.avg, epoch)
        writer.add_scalar('Test/Time', elapsed, epoch)

    # if config['tensorboard_parameters']:
    #     for name, param in model.named_parameters():
    #         writer.add_histogram(name, param, global_step)

    return angle_error_meter.avg




def main(opt):
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
    train_path = "../VOC/train.txt"
    val_path = "../VOC/val.txt"
    # train_path = "E:/GazeStudy/pytorch-yolo2-master/data/VOCtrainval_06-Nov-2007/2007_train.txt"
    # val_path = "E:/GazeStudy/pytorch-yolo2-master/data/VOCtrainval_06-Nov-2007/2007_val.txt"
    # optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    start_epoch = 0
    # print('Setting up data...')
    batchsize = 32 
    imageshape=(384,384)
    val_loader = torch.utils.data.DataLoader(
            listDataset(val_path, shape=imageshape,shuffle = False, 
            train=False,seen = 0,batch_size=batchsize), 
        
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        batch_size=batchsize, 
    ) 


    train_loader = torch.utils.data.DataLoader(
            listDataset(train_path, shape=imageshape,shuffle = True, 
            train=True,seen = 0,batch_size=batchsize),  
        batch_size=batchsize, 
        shuffle=True,
        num_workers=8,
        pin_memory=True,  
    )  
    print("the train_loader size is {}".format(len(train_loader)))

    writer = SummaryWriter()
    num_train_batches = int(len(train_loader))
    num_val_batches = int(len(val_loader))
    logger.info("the train batches is {}".format(num_train_batches))

    logger.info("the val batches is {}".format(num_val_batches))

    from models import get_pose_net
    heads = {"hm":20,"wh":2,"reg":2}
    model = get_pose_net(18,heads, head_conv=256)
    model.cuda()

    criterion = CtdetLoss(opt)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=opt.lr,
        momentum = 0.9,
        weight_decay=1e-4
        )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=opt.lr_step, gamma=0.1)

    config = {
        'tensorboard': True,
        'tensorboard_images': True,
        'tensorboard_parameters': True,
    }

    # run test before start training
    test(0, model, criterion, val_loader, config, writer)

    for epoch in range(1, opt.num__epoch):
        scheduler.step()

        train(epoch, model, optimizer, criterion, train_loader, config, writer)
        angle_error = test(epoch, model, criterion, val_loader, config,
                           writer)

        state = OrderedDict([
            # ('args', vars(args)),
            ('state_dict', model.state_dict()),
            ('optimizer', optimizer.state_dict()),
            ('epoch', epoch),
            ('angle_error', angle_error),
        ])
        outdir = "./"
        model_path = os.path.join(outdir, 'model_state1.pth')
        torch.save(state, model_path)

   # if args.tensorboard:
   #     outpath = os.path.join(outdir, 'all_scalars.json')
   #     writer.export_scalars_to_json(outpath)

      
if __name__ == '__main__':
    opt = opts().parse()
    main(opt)
