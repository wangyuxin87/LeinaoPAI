import numpy as np

from tqdm import tqdm
import shutil

import torch
from torch.backends import cudnn
from torch.autograd import Variable

from graphs.models import mtcnn
from dataloaders.mtcnn import MTCNNDataLoader
from graphs.losses.mtcnn_loss import MTCNNLoss

from torch.optim import lr_scheduler

from tensorboardX import SummaryWriter
from utils.metrics import AverageMeter
from utils.misc import print_cuda_statistics

from trainers.base import BaseTrainer

cudnn.benchmark = True


class MTCNNTrainer(BaseTrainer):
    """
    This class will be responsible for handling the whole process of our architecture.
    """

    def __init__(self, config):
        super().__init__(config)
        # Create an instance from the Model
        # self.logger.info("Loading encoder pretrained in imagenet...")
        # if self.config.pretrained_encoder:
        #     pretrained_enc = torch.nn.DataParallel(ERFNet(self.config.imagenet_nclasses)).cuda()
        #     pretrained_enc.load_state_dict(torch.load(self.config.pretrained_model_path)['state_dict'])
        #     pretrained_enc = next(pretrained_enc.children()).features.encoder
        # else:
        #     pretrained_enc = None
        # define erfNet model
        net = getattr(mtcnn, self.config.net)
        self.model = net(self.config)
        # Create an instance from the data loader
        self.data_loader = MTCNNDataLoader(self.config)
        # Create instance from the loss
        self.loss = MTCNNLoss()
        # Create instance from the optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        # self.optimizer = torch.optim.Adam(self.model.parameters(),
        #                                   lr=self.config.learning_rate,
        #                                   betas=(self.config.betas[0], self.config.betas[1]),
        #                                   eps=self.config.eps,
        #                                   weight_decay=self.config.weight_decay)
        # Define Scheduler
        lambda1 = lambda epoch: pow((1 - ((epoch - 1) / self.config.max_epoch)), 0.9)
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda1)
        # initialize my counters
        self.current_epoch = 0
        self.current_iteration = 0
        self.valid_loss = 1e4

        # Check is cuda is available or not
        self.is_cuda = torch.cuda.is_available()
        # Construct the flag and make sure that cuda is available
        self.cuda = self.is_cuda & self.config.cuda

        if self.cuda:
            torch.cuda.manual_seed_all(self.config.seed)
            self.device = torch.device("cuda")
            torch.cuda.set_device(self.config.gpu_device)
            self.logger.info("Operation will be on *****GPU-CUDA***** ")
            print_cuda_statistics()

        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.config.seed)
            self.logger.info("Operation will be on *****CPU***** ")

        self.model = self.model.to(self.device)
        # self.loss = self.loss.to(self.device)
        # Model Loading from the latest checkpoint if not found start from scratch.
        self.load_checkpoint(self.config.checkpoint_file)

        # Tensorboard Writer
        self.summary_writer = SummaryWriter(log_dir=self.config.summary_dir, comment='FCN8s')

        # # scheduler for the optimizer
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     self.optimizer, 'min', patience=self.config.learning_rate_patience, min_lr=1e-10, verbose=True)

    def save_checkpoint(self, filename='checkpoint.pth.tar', is_best=0):
        """
        Saving the latest checkpoint of the training
        :param filename: filename which will contain the state
        :param is_best: flag is it is the best model
        :return:
        """
        state = {
            'epoch': self.current_epoch + 1,
            'iteration': self.current_iteration,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        # Save the state
        torch.save(state, self.config.checkpoint_dir + filename)
        # If it is the best copy it to another file 'model_best.pth.tar'
        if is_best:
            shutil.copyfile(self.config.checkpoint_dir + filename,
                            self.config.checkpoint_dir + 'model_best.pth.tar')

    def load_checkpoint(self, filename):
        filename = self.config.checkpoint_dir + filename
        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)

            self.current_epoch = checkpoint['epoch']
            self.current_iteration = checkpoint['iteration']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

            self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
                             .format(self.config.checkpoint_dir, checkpoint['epoch'], checkpoint['iteration']))
        except OSError as e:
            self.logger.info("No checkpoint exists from '{}'. Skipping...".format(self.config.checkpoint_dir))
            self.logger.info("**First time to train**")

    def run(self):
        """
        This function will the operator
        :return:
        """
        assert self.config.mode in ['train', 'test', 'random']
        try:
            if self.config.mode == 'test':
                self.test()
            else:
                self.train()

        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def train(self):
        """
        Main training function, with per-epoch model saving
        """

        for epoch in range(self.current_epoch, self.config.max_epoch):
            self.current_epoch = epoch
            self.scheduler.step(epoch)
            self.train_one_epoch()

            valid_loss = self.validate()
            self.scheduler.step(valid_loss)

            is_best = valid_loss < self.valid_loss
            if is_best:
                self.valid_loss = valid_loss

            self.save_checkpoint(is_best=is_best)

    def train_one_epoch(self):
        """
        One epoch training function
        """
        # Initialize tqdm
        tqdm_batch = tqdm(self.data_loader.train_loader, total=self.data_loader.train_iterations,
                          desc="Epoch-{}-".format(self.current_epoch))

        # Set the model to be in training mode (for batchnorm)
        self.model.train()
        # Initialize your average meters
        epoch_loss = AverageMeter()
        epoch_clsloss = AverageMeter()
        epoch_locloss = AverageMeter()
        epoch_lanloss = AverageMeter()

        for image, gt_label, gt_bbox, gt_landmark in tqdm_batch:
            if self.cuda:
                image, gt_label, gt_bbox, gt_landmark = \
                    image.pin_memory().cuda(async=self.config.async_loading), \
                    gt_label.cuda(async=self.config.async_loading), \
                    gt_bbox.cuda(async=self.config.async_loading), \
                    gt_landmark.cuda(async=self.config.async_loading)
            image = Variable(image)
            gt_label = Variable(gt_label)
            gt_bbox = Variable(gt_bbox)
            gt_landmark = Variable(gt_landmark)
            # model
            if self.config.net == 'ONet':
                cls_pred, loc_preds, landmark_offset_pred = self.model(image)
                # loss
                cls_loss = self.loss.cls_loss(gt_label, cls_pred)
                loc_loss = self.loss.box_loss(gt_label, gt_bbox, loc_preds)
                lan_loss = self.loss.landmark_loss(gt_label, gt_landmark, landmark_offset_pred)
                cur_loss = cls_loss * 0.8 + loc_loss * 0.6 + lan_loss * 1.5
            else:
                cls_pred, loc_preds = self.model(image)
                # loss
                cls_loss = self.loss.cls_loss(gt_label, cls_pred)
                loc_loss = self.loss.box_loss(gt_label, gt_bbox, loc_preds)
                lan_loss = None
                cur_loss = cls_loss * 1.0 + loc_loss * 0.5

            if np.isnan(float(cur_loss.item())):
                raise ValueError('Loss is nan during training...')

            # optimizer
            self.optimizer.zero_grad()
            cur_loss.backward()
            self.optimizer.step()

            epoch_loss.update(cur_loss.item())
            epoch_clsloss.update(cls_loss.item())
            epoch_locloss.update(loc_loss.item())
            if self.config.net == 'ONet':
                epoch_lanloss.update(lan_loss.item())

            self.current_iteration += 1
            # exit(0)

        self.summary_writer.add_scalar("epoch-training/loss", epoch_loss.val, self.current_iteration)
        self.summary_writer.add_scalar("epoch-training/cls_loss", epoch_clsloss.val, self.current_iteration)
        self.summary_writer.add_scalar("epoch-training/loc_loss", epoch_locloss.val, self.current_iteration)
        if self.config.net == 'ONet':
            self.summary_writer.add_scalar("epoch-training/lan_loss", epoch_lanloss.val, self.current_iteration)
        tqdm_batch.close()

        print("Training Results at epoch-" + str(self.current_epoch) + " | " + "loss: " + str(
            epoch_loss.val))

    def validate(self):
        """
        One epoch validation
        :return:
        """
        tqdm_batch = tqdm(self.data_loader.valid_loader, total=self.data_loader.valid_iterations,
                          desc="Valiation at -{}-".format(self.current_epoch))

        # set the model in training mode
        self.model.eval()

        epoch_loss = AverageMeter()
        epoch_clsloss = AverageMeter()
        epoch_locloss = AverageMeter()
        epoch_lanloss = AverageMeter()

        for image, gt_label, gt_bbox, gt_landmark in tqdm_batch:
            if self.cuda:
                image, gt_label, gt_bbox, gt_landmark = \
                    image.pin_memory().cuda(async=self.config.async_loading), \
                    gt_label.cuda(async=self.config.async_loading), \
                    gt_bbox.cuda(async=self.config.async_loading), \
                    gt_landmark.cuda(async=self.config.async_loading)
            image = Variable(image)
            gt_label = Variable(gt_label)
            gt_bbox = Variable(gt_bbox)
            gt_landmark = Variable(gt_landmark)
            # model
            if self.config.net == 'ONet':
                cls_pred, loc_preds, landmark_offset_pred = self.model(image)
                # loss
                cls_loss = self.loss.cls_loss(gt_label, cls_pred)
                loc_loss = self.loss.box_loss(gt_label, gt_bbox, loc_preds)
                lan_loss = self.loss.landmark_loss(gt_label, gt_landmark, landmark_offset_pred)
                cur_loss = cls_loss * 0.8 + loc_loss * 0.6 + lan_loss * 1.5
            else:
                cls_pred, loc_preds = self.model(image)
                # loss
                cls_loss = self.loss.cls_loss(gt_label, cls_pred)
                loc_loss = self.loss.box_loss(gt_label, gt_bbox, loc_preds)
                lan_loss = None
                cur_loss = cls_loss * 1.0 + loc_loss * 0.5

            if np.isnan(float(cur_loss.item())):
                raise ValueError('Loss is nan during Validation.')

            epoch_loss.update(cur_loss.item())
            epoch_clsloss.update(cls_loss.item())
            epoch_locloss.update(loc_loss.item())
            if self.config.net == 'ONet':
                epoch_lanloss.update(lan_loss.item())

        self.summary_writer.add_scalar("epoch-validation/loss", epoch_loss.val, self.current_iteration)
        self.summary_writer.add_scalar("epoch-validation/cls_loss", epoch_clsloss.val, self.current_iteration)
        self.summary_writer.add_scalar("epoch-validation/loc_loss", epoch_locloss.val, self.current_iteration)
        if self.config.net == 'ONet':
            self.summary_writer.add_scalar("epoch-validation/lan_loss", epoch_lanloss.val, self.current_iteration)

        print("Validation Results at epoch-" + str(self.current_epoch) + " | " + "loss: " + str(
            epoch_loss.val))

        tqdm_batch.close()

        return epoch_loss.val

    def test(self):
        # TODO
        pass

    def finalize(self):
        """
        Finalize all the operations of the 2 Main classes of the process the operator and the data loader
        :return:
        """
        print("Please wait while finalizing the operation.. Thank you")
        self.save_checkpoint()
        self.summary_writer.export_scalars_to_json("{}all_scalars.json".format(self.config.summary_dir))
        self.summary_writer.close()
        self.data_loader.finalize()
