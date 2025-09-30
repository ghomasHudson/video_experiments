import argparse
import time
import copy
import torch
import wandb
import os
import random
import torch.optim as optim
import warnings

from models import load_model
from models_u import load_model_u
from models_diffusion import load_model_diff
from tools.dataset_tools import load_multimodal, load_simulation
from tools.model_tools import add_noise_and_clip, load_criterion, configure_amp
from tools.testing_tools import get_test_outputs, get_test_seq, log_results_to_wandb
from tools.output_tools import print_epoch_stats, get_path_incremental, print_summary, start_wandb
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.data import DataLoader
from torch.nn.functional import interpolate, l1_loss as l1
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIMLoss
from ocr import calculate_ocr_losses
import torch.nn.functional as F
import torchvision
from PIL import Image
from diffusers import DDPMScheduler
from torchvision import transforms


def get_args():
    parser = argparse.ArgumentParser()
    # Training args
    parser.add_argument('--job', type=str, default='train', help="job name for output & wandb logging")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--workers', default=0, type=int, help="number of workers for dataloader")
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--bs', default=64, type=int)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--wd', default=1e-5, type=float, help="weight decay")
    parser.add_argument('--noise', default=0., type=float, help="level gaussian noise added to training inputs")
    parser.add_argument('--loss', type=str, default='ms-ssim', help="See model_tools.load_criterion() for options")
    parser.add_argument('--ssim_win_size', default=3, type=int, help="window size for ssim loss")
    parser.add_argument('--reduction', type=str, default='none', help="Type of batch reduction for loss")
    parser.add_argument('--save', default=0, type=int, choices=[0, 1], help="save best checkpoint")
    parser.add_argument('--output_dir', default='saved', type=str)
    parser.add_argument('--train_per_epoch', default=None, type=int, help="max number of training samples per epoch")
    parser.add_argument('--val_per_epoch', default=None, type=int, help="max number of validation samples per epoch")
    # Dataset args
    parser.add_argument("--dataset", type=str, default='3d', help="Can be a path, or key in tools.get_path()")
    parser.add_argument('--img_type', type=str, default='greyscale')
    parser.add_argument('--img_size', default=64, type=int)
    parser.add_argument('--input_frames', type=int, default=0)
    parser.add_argument('--target_frames', type=int, default=0)
    parser.add_argument('--preload', default=0, type=int, choices=[0, 1], help="Preload dataset into memory")
    parser.add_argument('--from_saved', default=0, type=int, choices=[0, 1], help="Use dataset caching")
    parser.add_argument('--sample_frac', default=1, type=float, help="fraction of dataset to use")
    parser.add_argument('--sample_frac_val', default=1, type=float, help="fraction of validation dataset to use")
    parser.add_argument('--random_start', default=0, type=int, choices=[0, 1], help="random start frame for each seq")
    parser.add_argument('--augment', default=0, type=int, choices=[0, 1], help="apply data augmentation")
    # Model args #
    parser.add_argument('--model_name', default='patch_decoder', type=str)
    parser.add_argument('--model_path', type=str, default=None, help="path to load model checkpoint")
    parser.add_argument('--embed_factor', default=1, type=int, help="hiddem dim as a multiple of patch height")
    parser.add_argument('--head_dim', default=0, type=int)
    parser.add_argument('--heads', default=1, type=int)
    parser.add_argument('--layers', default=1, type=int)
    parser.add_argument('--decoding_layers', default=1, type=int)
    parser.add_argument('--patch_size', default=16, type=int)
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--channels', default=1, type=int, help="number of channels in input data")
    parser.add_argument('--max_input_len', default=20, type=int, help="Max input length for model training")
    parser.add_argument('--torch_attn', default=1, type=int)
    parser.add_argument('--registers', default=0, type=int)
    parser.add_argument('--clm', default=1, type=int, choices=[0, 1], help="inputs are targets shifted by 1")
    parser.add_argument('--clm_t', default=1, type=int, choices=[0, 1], help="inputs are targets shifted by 1")
    parser.add_argument('--amp', default='fp16', type=str, choices=['fp16', 'fp32', 'half', 'cpu'])
    # Testing args
    parser.add_argument('--plot', default=0, type=int, choices=[0, 1], help="log metrics to wandb server")
    parser.add_argument('--batch_loss', default=0, type=int, choices=[0, 1], help="log batch losses")
    parser.add_argument('--test_seqs', default=0, type=int, help="Number of sequences to test. 0 for none")
    parser.add_argument('--test_in_frames', default=5, type=int, help="Number of input frames when testing")
    parser.add_argument('--test_out_frames', default=1, type=int, help="Number of frames to generate")
    parser.add_argument('--do_ocr', default=0, type=int, choices=[0, 1], help="Run OCR on generated text")
    parser.add_argument('--max_context_len', default=20, type=int, help="Max context length for output gen: 0 == None")
    parser.add_argument('--plot_attn', default=0, type=int, choices=[0, 1], help="Plot attention maps (patch decoder)")
    parser.add_argument('--wandb_entity', default=None, type=str)
    parser.add_argument('--wandb_project', default=None, type=str)
    parser.add_argument('--wandb_run_id', default=None, type=str)

    return parser.parse_args()


class Main():
    def __init__(self, args=None):
        # collect training configuration arguments and set job name
        args = get_args() if not args else args

        # start / resume wandb logging
        args = start_wandb(args) if args.wandb_entity is not None else args

        # load training and valdiation data
        if args.dataset not in ['arc']:
            train_data = load_multimodal(split='train', **vars(args))
            val_data = load_multimodal(split='val', **vars(args))
        else:
            train_data = load_simulation(split='train', **vars(args))
            val_data = load_simulation(split='val', **vars(args))
        # print summary
        print_summary(args, len(train_data), len(val_data))

        # load args.test_seq number of test sequences if plotting enabled
        if args.plot and args.test_seqs > 0:
            test_batch = get_test_seq(val_data, args.test_seqs)

        # load image collaters for dataloader
        train_collate = ImageCollator(train=True, device=args.device, clm=args.clm, noise=args.noise, amp=args.amp, 
                                      dataset=args.dataset)
        val_collate = ImageCollator(train=False, device=args.device, clm=args.clm, amp=args.amp, dataset=args.dataset)

        # Load datasets into torch dataloaders
        train_dataloader = DataLoader(train_data, batch_size=args.bs, shuffle=True, num_workers=args.workers,
                                      collate_fn=train_collate, pin_memory=True)
        val_dataloader = DataLoader(val_data, batch_size=args.bs, shuffle=True, num_workers=args.workers,
                                    collate_fn=val_collate, pin_memory=True)

        # Set training precision
        self.amp_type, self.amp = configure_amp(args.amp)

        # initialize model
        if args.model_name == 'patch_decoder_diffusion':
            self.model = load_model_diff(args).to(args.device)
        elif args.model_name == 'patch_decoder_u' or args.model_name == 'patch_decoder_u-large':
            self.model = load_model_u(args).to(args.device)
        else:
            self.model = load_model(args).to(args.device)
        self.model.half() if self.amp_type == 'half' else self.model
        # load from checkpoint if model path provided
        if args.model_path is not None:
            print("Loading checkpoint...")
            self.model.load_state_dict(torch.load(args.model_path))

        # print model stats
        print(f"Trainable parameters:  {sum(p.numel() for p in self.model.parameters() if p.requires_grad)} \n")

        if args.model_path is not None:
            args.lr = 0.00001
            args.reduction = 'mean'
            
        # Set loss function, amp scaler, and optimizer
        self.criterion = load_criterion(args)
        self.val_ssim = SSIMLoss(data_range=1.0, reduction='elementwise_mean').to(args.device)
        self.scaler = torch.amp.GradScaler(args.device)
        eps = 1e-4 if args.amp == 'half' else 1e-8

        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr, eps=eps, weight_decay=args.wd,
                                     foreach=False, fused=True)
        # self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0)

        # create linear learning rate warmup scheduler if resuming
        if args.model_path is not None:
            self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=0.25, total_iters=1000)
        else:
            self.scheduler = None

        # create model save path
        if args.save:
            save_path = get_path_incremental(f'{args.output_dir}/models', args.job) + '.pt'
            print(f"Saving model checkpoints to: {save_path} \n")

        # create output path
        output_path = get_path_incremental(f'{args.output_dir}/training', args.job)

        # Initialize training stats
        train_stats = {'best_l1': float('inf'), 'best_ssim': -1}
        best_model_state = None
        new_model_state = False

        # Training Loops
        for epoch in range(args.epochs):
            # iterate output: {loss, l1, sl1, ssim, time}
            train_out = self.iterate(args, train_dataloader, train=True, max_samples=args.train_per_epoch)
            if (epoch % 5 == 0 and epoch > 0) or args.dataset == 'clevrer':
                # run validation every 5 epochs
                val_out = self.iterate(args, val_dataloader, train=False, do_ocr=args.do_ocr and epoch % 10 == 0,
                                       max_samples=args.val_per_epoch, target_frames=args.target_frames)

                test_time = None
                # save best epoch on best val loss
                if (args.loss in ['mse', 'l1', 'smoothl1'] and val_out['target_l1'] < train_stats['best_l1']) or \
                   (args.loss in ['ssim', 'ms-ssim'] and val_out['target_ssim'] > train_stats['best_ssim']):
                    # Update best stats
                    train_stats['best_l1'] = val_out['target_l1']
                    train_stats['best_ssim'] = val_out['target_ssim']

                    # save model on best validation update
                    if args.save:
                        best_model_state = copy.deepcopy(self.model.state_dict())
                        new_model_state = True

                    # test and plot on new best val
                    if args.test_seqs > 0 and args.plot:
                        if not os.path.exists(output_path):
                            os.makedirs(output_path)
                        # test model generation on test sequences and log to wandb
                        test_output = get_test_outputs(self.model, test_batch, output_path, **vars(args))
                        test_time = test_output['time']
                        log_results_to_wandb(test_output, commit=False) if args.wandb_entity is not None else None

                # Save every 5 epochs except 0
                if args.save and epoch % 10 == 0 and epoch > 0 and new_model_state:
                    print("Saving model state...")
                    torch.save(best_model_state, save_path)
                    new_model_state = False

                # Plot epoch stats to wandb
                if args.plot:
                    wandb.log({
                        'Epoch': epoch,
                        f'Train loss ({args.loss})': train_out['loss'],
                        'Train L1 (full)': train_out['l1'],
                        'Val L1': val_out['target_l1'],
                        'Val SSIM': val_out['target_ssim'],
                        'Val L1 (full)': val_out['l1'],
                        'Val SSIM (full)': val_out['ssim']},
                        commit=True)

                # printout epoch stats
                print_epoch_stats(epoch, args.epochs, train_out, val_out, test_time=test_time)

        # save best model state at end of training
        if args.save:
            print("Saving final model state...")
            torch.save(best_model_state, save_path)

        print("Training ended.")
        print(train_stats)

    def iterate(self, args, loader, train=True, clm=True, target_frames=0, do_ocr=False, max_samples=None):
        # Initialise epoch metrics dictionary
        start = time.time()
        epoch_stats = {'loss': 0, 'l1': 0, 'ssim': 0, 'target_l1': 0, 'target_ssim': 0}
        total_samples = 0

        # Set model to train or eval mode
        self.model.train() if train else self.model.eval()
        # Iteration loop over batches with gradients set if training
        with torch.set_grad_enabled(train):
            for idx, batch in enumerate(loader):
                # Unpack batch dictionary and send to device
                inputs = batch['inputs'].to(device=args.device, non_blocking=True)
                targets = batch['targets'].to(device=args.device, non_blocking=True)
                # padding_mask = batch['padding_mask'].to(device=args.device, non_blocking=True)

                # Get input shape
                b, t, c, h, w = inputs.size()
                num_targets = targets.size(1)

                if args.dataset == 'clevrer' and clm:
                    # remove last two frames from inputs and targets
                    inputs = inputs[:, :-2, ...].reshape(b, t - 2, c, h, w).contiguous()
                    targets = targets[:, :-2, ...].reshape(b, t - 2, c, h, w).contiguous()
                    b, t, c, h, w = inputs.size()
                    num_targets = targets.size(1)

                # Zero model gradients before each batch
                self.optimizer.zero_grad(set_to_none=True)

                # Run trainin mode with backprop
                if train:
                    self.model.reset_cache()
                    # Get model outputs, metrics, and update network if in train mode
                    with torch.autocast(args.device, dtype=self.amp_type, enabled=bool(self.amp)):
                        # Get model predictions
                        if args.model_name == 'patch_decoder_diffusion':
                            diff_t = torch.randint(0, 1000, (b * t,)).to(targets.device).long()
                            out = self.model(inputs, diff_t, targets=targets.clone())
                            predictions = out['predictions']
                            target_noise = out['target_noise']
                        elif args.model_name == 'patch_decoder_u':
                            out = self.model(inputs, targets=None)
                            predictions = out['predictions']
                        else:
                            out = self.model(inputs, targets=targets.clone())
                            predictions = out['predictions']  # Expected predictions tensor shape: (b, t, c, h, w)

                        # If causal language modeling (clm) not set, get len target frames from predictions
                        if not clm:
                            predictions = predictions[:, -num_targets:, ...]

                        if args.clm_t is True:
                            predictions = predictions[:, -target_frames:, ...]
                            targets = targets[:, -target_frames:, ...]

                        # Fold timesteps onto batch dimension for loss calculations
                        predictions = predictions.reshape(-1, c, h, w)
                        targets = targets.reshape(-1, c, h, w)

                        if args.model_name == 'patch_decoder_diffusion':
                            target_noise = target_noise.reshape(-1, c, h, w)
                            loss = F.mse_loss(predictions, target_noise, reduction=args.reduction)

                        # Calculate loss and update model when train flagged
                        elif args.loss in ['ssim', 'ms-ssim']:
                            # Calculate SSIM loss on float32 tensors
                            with torch.autocast(args.device, dtype=torch.float32, enabled=False):
                                loss = self.criterion(predictions.float(), targets.float())

                        else:
                            loss = self.criterion(predictions, targets)

                        # Get gradient tensor if loss reduction is none
                        grad = torch.ones_like(loss) if args.reduction == 'none' else None

                        # # Network update
                        if self.amp:
                            self.scaler.scale(loss).backward(grad)
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            loss.backward(grad)
                            self.optimizer.step()

                        # Update scheduler if set
                        if self.scheduler is not None:
                            self.scheduler.step()

                        # Get batch losses and plot if set
                        batch_loss = loss.mean().detach()
                        epoch_stats['loss'] += batch_loss
                        l1_loss = l1(predictions, targets).detach().item()
                        epoch_stats['l1'] += l1_loss

                        # Plot batch losses to wandb
                        if args.plot and args.batch_loss:
                            wandb.log({f'Batch loss ({args.loss})': batch_loss})

                # Run validation mode and calculate metrics
                elif not train:
                    with torch.autocast(args.device, dtype=self.amp_type, enabled=bool(self.amp)):
                        if args.model_name == 'patch_decoder_diffusion':
                            out = self.model(inputs, 0)
                        else:
                            out = self.model(inputs, cache_decoder_attn=True)
                        self.model.reset_cache()
                        predictions = out['predictions']  # Expected predictions tensor shape: (b, t, c, h, w)

                        # If causal language modeling not set, get len target frames from predictions
                        if not clm:
                            predictions = predictions[:, -num_targets:, ...]

                        predictions = predictions.view(-1, c, h, w)
                        targets = targets.view(-1, c, h, w)

                        # Calculate l1
                        epoch_stats['l1'] += l1(predictions, targets).detach().item()

                        # Calculate SSIM loss on float32 tensors
                        with torch.autocast(args.device, dtype=torch.float32, enabled=False):
                            epoch_stats['ssim'] += self.val_ssim(predictions.float(), targets.float()).detach().item()

                        predictions = predictions.view(b, t, c, h, w)[:, -target_frames:, ...].reshape(-1, c, h, w)
                        targets = targets.view(b, t, c, h, w)[:, -target_frames:, ...].reshape(-1, c, h, w)

                        # Calculate l1
                        epoch_stats['target_l1'] += l1(predictions, targets).detach().item()

                        # Calculate SSIM loss on float32 tensors
                        with torch.autocast(args.device, dtype=torch.float32, enabled=False):
                            epoch_stats['target_ssim'] += self.val_ssim(predictions.float(),
                                                                        targets.float()).detach().item()

                    # calculate OCR losses if set
                    if do_ocr:
                        try:
                            ocr_losses = calculate_ocr_losses(batch, predictions, targets,
                                                              vocab=loader.dataset.vocab(), limit=1)
                            for k, v in ocr_losses.items():
                                if k not in epoch_stats:
                                    epoch_stats[k] = 0
                                epoch_stats[k] += v
                        except Exception as e:
                            print("OCR Error:", e)

                # Break loop if max_samples set
                total_samples += b
                if max_samples is not None and total_samples >= max_samples:
                    break

        # Average accumulated metrics over num of batches
        epoch_stats = dict((k, v / (idx + 1)) for k, v in epoch_stats.items())
        epoch_stats['time'] = time.time() - start

        return epoch_stats


class ImageCollator(object):
    def __init__(self, train=False, device='cuda', clm=True, noise=0, amp='fp16', padding_side='left', dataset=None,
                 img_size=64):
        self.train = train
        self.device = device
        self.dataset = dataset
        self.noise = float(noise)
        self.clm = clm
        self.amp = amp
        self.padding_side = padding_side
        self.norm = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        self.flip = transforms.RandomHorizontalFlip()
        self.crop = transforms.RandomCrop(size=img_size, padding=8, 
                                          padding_mode='constant')

    def __call__(self, batch):
        # check if batch items are of different lengths and left-pad if so
        max_len = max([x['inputs'].shape[0] for x in batch])

        with torch.no_grad():
            # Process batch items

            padding_mask = torch.zeros(len(batch), max_len)
            for idx, _ in enumerate(batch):
                # convert inputs to torch tensors if not already
                if not isinstance(batch[idx]['inputs'], torch.Tensor):
                    batch[idx]['inputs'] = torch.from_numpy(batch[idx]['inputs']).float()

                # convert targets to torch tensors if not already
                if batch[idx]['targets'] is not None and not isinstance(batch[idx]['targets'], torch.Tensor):
                    batch[idx]['targets'] = torch.from_numpy(batch[idx]['targets']).float()

                # Pad inputs to max length with -1
                if batch[idx]['inputs'].shape[0] < max_len:
                    # create a pad tensor with same shape as inputs at batch idx minus time dimension
                    pad = torch.zeros(max_len - batch[idx]['inputs'].shape[0], *batch[idx]['inputs'].shape[1:]) - 1
                    if self.padding_side == 'left':
                        batch[idx]['inputs'] = torch.cat([pad, batch[idx]['inputs']])
                    elif self.padding_side == 'right':
                        batch[idx]['inputs'] = torch.cat([batch[idx]['inputs'], pad])

                # If last dimension size is 1 or 3 (assuming this is the colour channel), swap axes to [b, t, c, h, w]
                if batch[idx]['inputs'].shape[-1] in [1, 3]:
                    batch[idx]['inputs'] = batch[idx]['inputs'].permute(0, 3, 1, 2)

                # If targets are not None and last dimension size is 1 or 3, swap axes to [b, t, c, h, w]
                if batch[idx]['targets'] is not None and batch[idx]['targets'].shape[-1] in [1, 3]:
                    batch[idx]['targets'] = batch[idx]['targets'].permute(0, 3, 1, 2)

                # Detect padding images and store to mask tensor
                # For any image with pixel value of -1, set corresponding timestep in padding_mask to 1
                padding_mask[idx, :max_len] = (batch[idx]['inputs'] == -1).any(dim=(1, 2, 3))

            # convert padding tensor to boolean tensor
            padding_mask = padding_mask.bool()

            # Stack inputs and targets into tensors
            inputs = torch.stack([x['inputs'] for x in batch]).float()
            if batch[0]['targets'] is not None:
                targets = torch.stack([x['targets'] for x in batch]).float()
            else:
                targets = None

            # get sequence names if provided
            try:
                sequence_name = [x['sequence_name'] for x in batch]
            except KeyError:
                sequence_name = None

            # get config if provided
            try:
                configs = [x['config'] for x in batch]
            except KeyError:
                configs = None

            # if inputs / targets in 0-255 range, scale to 0-1
            if inputs.max() > 1:
                inputs = inputs / 255
            if targets is not None and targets.max() > 1:
                targets = targets / 255
                
            if self.train:
                # apply transforms to inputs
                for i in range(inputs.size(0)):
                    if random.random() > 0.5 and self.dataset in ['colorization', 'cifar10', 'lasot']:
                        inputs[i] = self.flip(inputs[i])
                    if random.random() > 0.5 and self.dataset in ['colorization', 'cifar10', 'lasot', 
                                                                  'tinyvirat']:
                        inputs[i] = self.crop(inputs[i])
                    # if random.random() > 0.5 and self.dataset not in ['colorization']:
                    #     inputs[i] = self.jitter(inputs[i])
                        
            # if clm set, right-shift targets by 1
            if self.clm:
                targets = inputs[:, 1:, ...].clone().detach()
                inputs = inputs[:, :-1, ...]
                padding_mask = padding_mask[:, :-1]

            # if targets is None and no CLM set, set to last frame of inputs
            elif targets is None:
                targets = inputs[:, -1:, ...].clone().detach()
                inputs = inputs[:, :-1, ...]
                padding_mask = padding_mask[:, :-1]

            # Create single dummy colour channel for greyscale images
            if len(inputs.size()) == 4:
                inputs = inputs.unsqueeze(2)
                targets = targets.unsqueeze(2)
            elif len(inputs.size()) != 5:
                raise ValueError(f"Input tensor has incorrect dimensions: {inputs.size()}. Requires 4 or 5.")

            # add noise to inputs if set
            if self.train and self.noise != 0:
                inputs = add_noise_and_clip(inputs, mean=0, std_range=(0, 0.1), mask_percent=0.2).float()
                
            if self.amp == 'half':
                inputs = inputs.half()
                targets = targets.half() if targets is not None else None
            
        return {'inputs': inputs, 'targets': targets, 'padding_mask': padding_mask, 'sequence_name': sequence_name,
                'config': configs}


if __name__ == '__main__':
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    Main()
