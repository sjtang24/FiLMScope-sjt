from filmscope.datasets import FSDataset
from filmscope.models import VolumeConvNet
from filmscope.losses import UnSupLoss
from filmscope.config import path_to_data
from filmscope.recon_util import (tocuda, get_ss_volume_from_dataset,
                                  get_height_aware_vol_from_dataset)
from .log_manager import NeptuneLogManager
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import time 

class RunManager:
    def __init__(self, timing_dict, config_dict, sample=None, calibration_file=None, guide_map=None,
                 prev_model=None, global_mask=None, run_name=None):
        # sample should be the xarray passed in by the mcam loop
        self.run_name = run_name
        self.config_dict = config_dict
        self.timing_dict = timing_dict
        self.timing_dict['setup_inner'] = {}
        self.run_args = config_dict["run_args"]
        self.info = config_dict["sample_info"]
        self.loss_w = config_dict["loss_weights"]

        self.guide_map = guide_map
        if self.guide_map is not None: 
            self.guide_map = self.guide_map.cuda()
        self.global_mask = global_mask
        if self.global_mask is not None:
            self.global_mask = self.global_mask.cuda()

        frame_number = self.run_args["frame_number"]

        if 'blank_filename' in self.info and self.info["blank_filename"] is not None:
            blank_filename = path_to_data + self.info["blank_filename"]
        else:
            blank_filename = None
        self.setup_time_per_frame = []
        self.times_per_frame = []
        self.frame_time = None
     
        if prev_model is not None:
            self.model = prev_model
        else:
            self.model = VolumeConvNet(
                num_channels=1,
                num_layers=self.run_args["unet_layers"],
                layer_channels=self.run_args["unet_layer_channels"],
                layer_strides=self.run_args["unet_layer_strides"]
            ).cuda()

        assert self.run_args["optim"] == "Adam"
        optim_params = self.model.parameters() 
        self.optimizer = optim.Adam(
            optim_params,
            lr=self.run_args["lr"],
            betas=(0.9, 0.999),
            weight_decay=self.run_args["weight_decay"]
        )

        # make the criterion
        self.criterion = UnSupLoss(
            smooth_weight=self.loss_w["smooth"],
            ssim_weight=self.loss_w["ssim"],
            smooth_lambda=self.loss_w["smooth_lambda"]
        ).cuda()

        if sample is None:
            image_filename = path_to_data + self.info["image_filename"]
        else:
            image_filename = None 

        #torch.cuda.synchronize()
        #start_load = time.perf_counter()
        self.dataset = FSDataset(
            self.timing_dict,
            calibration_file, #path_to_data + self.info["calibration_filename"],
            self.info["image_numbers"],
            sample = sample, image_filename=image_filename, #path_to_data + self.info["image_filename"]
            downsample=self.info["downsample"],
            crop_values=self.info["crop_values"],
            frame_number=frame_number,
            ref_crop_center=self.info["ref_crop_center"],
            crop_size=self.info["crop_size"],
            height_est=self.info["height_est"],
            blank_filename=blank_filename,
        )
        #torch.cuda.synchronize()
        #end_load = time.perf_counter()
        #self.timing_dict['setup_inner']['create-dataset'] = [end_load - start_load]

        #torch.cuda.synchronize()
        #start_dler = time.perf_counter()
        self.image_loader = DataLoader(
            self.dataset,
            self.run_args["batch_size"],
            shuffle=self.run_args["loader_shuffle"],
            num_workers=self.run_args["loader_num_workers"],
            drop_last=self.run_args["drop_last"],
            pin_memory=True
        )
        #torch.cuda.synchronize()
        #end_dler = time.perf_counter()
        #self.timing_dict['setup_inner']['batching'] = [end_dler - start_dler]

        #torch.cuda.synchronize()
        #start_transfer = time.perf_counter()
        # prepare other things needed throughout reconstruction
        self.reference_image = self.dataset.reference_image.cuda()
        self.reference_shift_slopes = self.dataset.ref_camera_shift_slopes.cuda()
        self.depth_values = torch.linspace(
                self.info["depth_range"][0],
                self.info["depth_range"][1],
                self.run_args["num_depths"], dtype=torch.float32)
        self.depth_values = tocuda(self.depth_values)
        #torch.cuda.synchronize()
        #end_transfer = time.perf_counter()
        #transfer_duration = end_transfer - start_transfer

        #torch.cuda.synchronize()
        #start_prepvolume = time.perf_counter()
        self.prepare_volume()
        #torch.cuda.synchronize()
        #end_prepvolume = time.perf_counter()
        #self.timing_dict['setup_inner']['prep-volume'] = end_prepvolume - start_prepvolume
        
        #torch.cuda.synchronize()
        #start_transfer1 = time.perf_counter()
        self.dataset.to_device("cuda")
        #torch.cuda.synchronize()  # Ensure GPU is done
        #end_transfer1 = time.perf_counter()
        #transfer_duration += (end_transfer1 - start_transfer1)
        #self.timing_dict['setup_inner']['transfer-gpu'] = transfer_duration

        self.logger = None
        if config_dict["use_neptune"]:
            self.setup_logger()

    def prepare_volume(self):
        with torch.no_grad():
            if self.guide_map is None:
                volume, volume_sq = get_ss_volume_from_dataset(
                    self.dataset,
                    self.run_args["batch_size"],
                    self.depth_values,
                    get_squared=True,
                )
            else:
                volume, volume_sq = get_height_aware_vol_from_dataset(
                    self.dataset,
                    self.run_args["batch_size"],
                    self.depth_values,
                    self.guide_map,
                    get_squared=True
                )
            num_views = len(self.dataset)
            self.volume_variance = volume_sq.div_(num_views).sub_(
                volume.div_(num_views).pow_(2)
            )

    def setup_logger(self): 
        self.logger = NeptuneLogManager(
            dataset=self.dataset,
            model=self.model,
            config_dictionary=self.config_dict, 
            run_name=self.run_name
        )

    # TODO: create a new function to pass in an xarray

    def swap_frames(self, frame_number = -1, sample_image = None):
        if 'swap_info' not in self.timing_dict:
            self.timing_dict['swap_info'] = {
                'transfer-cpu' : [],
                'transfer-gpu' : [],
                'swap-frames' : [],
                'prep-volume' : []
            }

        self.run_args["frame_number"] = frame_number

        # not sure why this is necessary
        # the the DataLoader used to make the volume 
        # fails if the data is not moved back to the cpu
        #torch.cuda.synchronize()
        #start_cpu_transfer = time.perf_counter()
        self.dataset.to_device("cpu")
        #torch.cuda.synchronize()
        #end_cpu_transfer = time.perf_counter()
        #self.timing_dict['swap_info']['transfer-cpu'].append(end_cpu_transfer - start_cpu_transfer)

        #torch.cuda.synchronize()
        #start_swap = time.perf_counter()
        self.dataset.swap_frames(frame_number = frame_number, sample_image = sample_image)
        #torch.cuda.synchronize()
        #end_swap = time.perf_counter()
        #self.timing_dict['swap_info']['swap-frames'].append(end_swap - start_swap)

        self.timing_dict = self.dataset.timing_dict

       # torch.cuda.synchronize()
        #start_gpu_transfer = time.perf_counter()
        self.reference_image = self.dataset.reference_image.cuda()
        #torch.cuda.synchronize()
        #end_gpu_transfer = time.perf_counter()
        #gpu_transfer_time = end_gpu_transfer - start_gpu_transfer

        #torch.cuda.synchronize()
        #start_prepvol = time.perf_counter()
        self.prepare_volume()
        #torch.cuda.synchronize()
        #end_prepvol = time.perf_counter()
        #self.timing_dict['swap_info']['prep-volume'].append(end_prepvol - start_prepvol)

        #torch.cuda.synchronize()
        #start_gpu_transfer2 = time.perf_counter()
        self.dataset.to_device("cuda")
        #torch.cuda.synchronize()  # Ensure GPU is done
        #end_gpu_transfer2 = time.perf_counter()
        #gpu_transfer_time += (end_gpu_transfer2 - start_gpu_transfer2)
        #self.timing_dict['swap_info']['transfer-gpu'].append(gpu_transfer_time)

        if self.config_dict["use_neptune"]:
            self.setup_logger()

    def run_forward_model(self):
        outputs = {}
        outputs["depth"]= self.model(
            self.volume_variance, 
            self.depth_values,
        )

        if self.guide_map is not None:
            # add the guide map to the depth 
            outputs["depth"] = outputs["depth"] + self.guide_map.squeeze(-1)
        return outputs
    
    def train_sample(self, sample):
        self.model.train()
        self.optimizer.zero_grad()

        outputs = self.run_forward_model()

        # compute the loss
        loss_values = {}
        loss_outputs = {}

        main_loss, losses, loss_outputs = self.criterion(
            sample["imgs"],
            outputs["depth"],
            sample["warped_shift_slope_maps"],
            sample["inv_inter_camera_maps"],
            self.reference_shift_slopes,
            self.reference_image,
            sample["masks"],
            global_mask=self.global_mask,
        )

        for key, item in losses.items():
            loss_values[key] = item
        for key, item in loss_outputs.items():
            loss_outputs[key] = item

        loss_values["total"] = main_loss
        main_loss.backward()
        self.optimizer.step()

        for key in ["warped_imgs", "masks"]:
            outputs[key] = loss_outputs[key]
        return outputs, loss_values

    def run_epoch(self, i, log=False):
        sample = self.dataset.get_full_sample()
        numbers = sample['image_numbers'].tolist()
        outputs, loss_values = self.train_sample(sample)

        warp_images = outputs["warped_imgs"]
        mask_images = outputs["masks"]

        if self.logger is not None:
            self.logger.log_loss(loss_values)
        
        if log:
            if self.logger is None:
                print("only set up for logging with Neptune")
            else:
                self.log_results(mask_images, warp_images, outputs, i) 

        return mask_images, warp_images, numbers, outputs, loss_values 

    def log_results(self, mask_images, warp_images, outputs, epoch):
        # log the model 
        self.logger.log_model(self.model, epoch)

        # log the summed masks and warped images
        sum_mask = torch.mean(mask_images.to(torch.float32), axis=0).squeeze().cpu().detach()
        self.logger.log_values(sum_mask, "summed_mask")
        sum_warp = torch.mean(warp_images, axis=0).squeeze().cpu().detach()
        self.logger.log_values(sum_warp, "summed_warp")

        # log depth as a plot and raw data
        depth = outputs["depth"].squeeze().cpu().detach()
        self.logger.log_values(depth, "depth")
        self.logger.log_value_plot(depth, "depth", epoch, cmap='turbo')
        return

    def end(self):
        self.__del__()

    def __del__(self):
        if self.logger is not None:
            self.logger.end()