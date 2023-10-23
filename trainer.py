import os
import math
from time import time
from tqdm import tqdm
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_
from recbole.trainer import Trainer
from recbole.utils import set_color, get_gpu_usage, EvaluatorType, early_stopping, dict2str, get_local_time
from recbole.data.dataloader import FullSortEvalDataLoader
import torch.distributed as dist

from cluster_utils import cluster_dpc_knn, cluster_kmeans


class MISSRecTrainer(Trainer):
    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        r"""Train the model in an epoch

        Args:
            train_data (DataLoader): The train data.
            epoch_idx (int): The current epoch id.
            loss_func (function): The loss function of :attr:`model`. If it is ``None``, the loss function will be
                :attr:`self.model.calculate_loss`. Defaults to ``None``.
            show_progress (bool): Show the progress of training epoch. Defaults to ``False``.

        Returns:
            float/tuple: The sum of loss returned by all batches in this epoch. If the loss in each batch contains
            multiple parts and the model return these multiple parts loss instead of the sum of loss, it will return a
            tuple which includes the sum of loss in each part.
        """
        self._multimodal_interest_discovery(epoch_idx)
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        iter_data = (
            tqdm(
                train_data,
                total=len(train_data),
                ncols=100,
                desc=set_color(f"Train {epoch_idx:>5}", 'pink'),
            ) if show_progress else train_data
        )
        for batch_idx, interaction in enumerate(iter_data):
            interaction = interaction.to(self.device)
            self.optimizer.zero_grad()
            losses = loss_func(interaction)
            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
            self._check_nan(loss)
            loss.backward()
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            self.optimizer.step()
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(set_color('GPU RAM: ' + get_gpu_usage(self.device), 'yellow'))
        return total_loss

    @torch.no_grad()
    def _multimodal_interest_discovery(self, epoch_idx=None):
        self.logger.info(f'Discovering multi-modal user interest' + ('' if epoch_idx is None else f' before {epoch_idx}-th epoch'))
        all_embedding_list = []
        self.model.eval()
        # get modality embeddings
        if 'text' in self.model.modal_type:
            all_text_embeddings = self.model.text_adaptor(self.model.plm_embedding.weight.data[1:])
            all_embedding_list.append(all_text_embeddings)
        if 'img' in self.model.modal_type:
            all_img_embeddings = self.model.img_adaptor(self.model.img_embedding.weight.data[1:])
            all_embedding_list.append(all_img_embeddings)
        all_embedding = torch.cat(all_embedding_list, dim=0)
        
        # multi-modal interest discovery
        # cluster_idx, centroids = cluster_dpc_knn(
        cluster_idx, centroids = cluster_kmeans(
            all_embedding, 
            cluster_num=self.model.num_interest, 
            # local_size=self.model.knn_local_size
        )
        cluster_idx += 1 # offset 1 for padding_idx
        
        # write back to the model
        self.model.interest_embeddings[1:] = centroids
        read_ptr = 0
        if 'text' in self.model.modal_type:
            self.model.plm_interest_lookup_table[1: 1+len(all_text_embeddings)] = cluster_idx[read_ptr: read_ptr+len(all_text_embeddings)]
            read_ptr += len(all_text_embeddings)
        if 'img' in self.model.modal_type:
            self.model.img_interest_lookup_table[1: 1+len(all_img_embeddings)] = cluster_idx[read_ptr: read_ptr+len(all_img_embeddings)]
            read_ptr += len(all_img_embeddings)
        assert read_ptr == len(cluster_idx), f"read_ptr={read_ptr}, len(cluster_idx)={len(cluster_idx)}"
        del all_embedding_list, all_embedding, cluster_idx, centroids
        self.logger.info(f'Finish multi-modal interest discovery' + ('' if epoch_idx is None else f' before {epoch_idx}-th epoch'))


class DDPPretrainTrainer(Trainer):
    def __init__(self, config, model):
        super(DDPPretrainTrainer, self).__init__(config, model)
        self.pretrain_epochs = self.config['pretrain_epochs']
        self.save_step = self.config['save_step']
        self.rank = config['rank']
        self.world_size = config['world_size']
        self.lrank = self._build_distribute(rank=self.rank, world_size=self.world_size)
        self.logger.info(f'Let\'s use {torch.cuda.device_count()} GPUs to train {self.config["model"]} ...')
        dist.barrier() # sync the timestamp to avoid inconsistency of checkpoint names
        saved_model_file = '{}-{}.pth'.format(self.config['model'], get_local_time())
        self.saved_model_file = os.path.join(self.checkpoint_dir, saved_model_file)

    def _build_distribute(self, rank, world_size):
        from torch.nn.parallel import DistributedDataParallel
        # credit to @Juyong Jiang
        # 1 set backend
        torch.distributed.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        # 2 get distributed id
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        device_dis = torch.device("cuda", local_rank)
        # 3, 4 assign model to be distributed
        self.model.to(device_dis)
        self.model = DistributedDataParallel(self.model, 
                                             device_ids=[local_rank],
                                             output_device=local_rank).module
        return local_rank

    def save_pretrained_model(self, epoch, saved_model_file):
        r"""Store the model parameters information and training information.

        Args:
            epoch (int): the current epoch id
            saved_model_file (str): file name for saved pretrained model

        """
        state = {
            'config': self.config,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(state, saved_model_file)

    def _trans_dataload(self, interaction):
        from torch.utils.data import DataLoader
        from torch.utils.data.distributed import DistributedSampler

        #using pytorch dataload to re-wrap dataset
        def sub_trans(dataset):
            dis_loader = DataLoader(dataset=dataset,
                                    batch_size=dataset.shape[0],
                                    sampler=DistributedSampler(dataset, shuffle=False))
            for data in dis_loader:
                batch_data = data

            return batch_data
        #change `interaction` datatype to a python `dict` object.  
        #for some methods, you may need transfer more data unit like the following way.  

        data_dict = {}
        for k, v in interaction.interaction.items():
            data_dict[k] = sub_trans(v)
        return data_dict

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        iter_data = (
            tqdm(
                train_data,
                total=len(train_data),
                ncols=100,
                desc=set_color(f"Train {epoch_idx:>5}", 'pink'),
            ) if show_progress else train_data
        )
        for batch_idx, interaction in enumerate(iter_data):
            interaction = interaction.to(self.device)
            interaction = self._trans_dataload(interaction)
            self.optimizer.zero_grad()
            losses = loss_func(interaction)
            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
            self._check_nan(loss)
            loss.backward()
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            self.optimizer.step()
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(set_color('GPU RAM: ' + get_gpu_usage(self.device), 'yellow'))
        return total_loss

    def pretrain(self, train_data, verbose=True, show_progress=False):
        for epoch_idx in range(self.start_epoch, self.pretrain_epochs):
            # train
            training_start_time = time()
            train_loss = self._train_epoch(train_data, epoch_idx, show_progress=show_progress)
            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
            if verbose:
                self.logger.info(train_loss_output)
            self._add_train_loss_to_tensorboard(epoch_idx, train_loss)

            if (epoch_idx + 1) % self.save_step == 0 and self.lrank == 0:
                saved_model_file = os.path.join(
                    self.checkpoint_dir,
                    '{}-{}-{}-{}.pth'.format(self.config['model'], self.config['dataset'], str(epoch_idx + 1),
                                             self.config['ckpt_suffix'] if 'ckpt_suffix' in self.config else 'paper')
                )
                self.save_pretrained_model(epoch_idx, saved_model_file)
                update_output = set_color('Saving current', 'blue') + ': %s' % saved_model_file
                if verbose:
                    self.logger.info(update_output)

        return self.best_valid_score, self.best_valid_result


class DDPTrainer(Trainer):
    def __init__(self, config, model):
        super(DDPTrainer, self).__init__(config, model)
        self.rank = config['rank']
        self.world_size = config['world_size']
        print("DDP TRAINER world_size", self.world_size, torch.cuda.device_count())
        self.lrank = self._build_distribute(rank=self.rank, world_size=self.world_size)
        self.logger.info(f'Let\'s use {torch.cuda.device_count()} GPUs to train {self.config["model"]} ...')
        dist.barrier() # sync the timestamp to avoid inconsistency of checkpoint names
        saved_model_file = '{}-{}.pth'.format(self.config['model'], get_local_time())
        self.saved_model_file = os.path.join(self.checkpoint_dir, saved_model_file)

    def _build_distribute(self, rank, world_size):
        from torch.nn.parallel import DistributedDataParallel
        # credit to @Juyong Jiang
        # 1 set backend
        torch.distributed.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        # 2 get distributed id
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        device_dis = torch.device("cuda", local_rank)
        # 3, 4 assign model to be distributed
        self.model.to(device_dis)
        self.model = DistributedDataParallel(self.model,
                                             device_ids=[local_rank],
                                             output_device=local_rank).module
        return local_rank

    def _trans_dataload(self, interaction):
        from torch.utils.data import DataLoader
        from torch.utils.data.distributed import DistributedSampler

        #using pytorch dataload to re-wrap dataset
        def sub_trans(dataset):
            if type(dataset) == torch.Tensor:
                dis_loader = DataLoader(dataset=dataset,
                                        batch_size=dataset.shape[0],
                                        sampler=DistributedSampler(dataset, shuffle=False))
            else:
                def collate_fn(batch):
                    return_batch = []
                    # Batching by use a list for non-fixed size
                    for value in batch:
                        return_batch.append(value)
                    return return_batch
                dis_loader = DataLoader(dataset=dataset,
                                        batch_size=dataset.shape[0],
                                        sampler=DistributedSampler(dataset, shuffle=False), collate_fn=collate_fn)

            for data in dis_loader:
                batch_data = data

            return batch_data
        #change `interaction` datatype to a python `dict` object.
        #for some methods, you may need transfer more data unit like the following way.

        data_dict = {}
        for k, v in interaction.interaction.items():
            data_dict[k] = sub_trans(v)
        return data_dict

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        r"""Train the model in an epoch

        Args:
            train_data (DataLoader): The train data.
            epoch_idx (int): The current epoch id.
            loss_func (function): The loss function of :attr:`model`. If it is ``None``, the loss function will be
                :attr:`self.model.calculate_loss`. Defaults to ``None``.
            show_progress (bool): Show the progress of training epoch. Defaults to ``False``.

        Returns:
            float/tuple: The sum of loss returned by all batches in this epoch. If the loss in each batch contains
            multiple parts and the model return these multiple parts loss instead of the sum of loss, it will return a
            tuple which includes the sum of loss in each part.
        """
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        iter_data = (
            tqdm(
                train_data,
                total=len(train_data),
                ncols=100,
                desc=set_color(f"Train {epoch_idx:>5}", 'pink'),
            ) if show_progress else train_data
        )
        for batch_idx, interaction in enumerate(iter_data):
            interaction = interaction.to(self.device)
            interaction = self._trans_dataload(interaction)
            self.optimizer.zero_grad()
            losses = loss_func(interaction)

            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
            self._check_nan(loss)
            loss.backward()
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            self.optimizer.step()
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(set_color('GPU RAM: ' + get_gpu_usage(self.device), 'yellow'))
        return total_loss

    @torch.no_grad()
    def evaluate(self, eval_data, load_best_model=True, model_file=None, show_progress=False):
        r"""Evaluate the model based on the eval data.

        Args:
            eval_data (DataLoader): the eval data
            load_best_model (bool, optional): whether load the best model in the training process, default: True.
                                              It should be set True, if users want to test the model after training.
            model_file (str, optional): the saved model file, default: None. If users want to test the previously
                                        trained model file, they can set this parameter.
            show_progress (bool): Show the progress of evaluate epoch. Defaults to ``False``.

        Returns:
            collections.OrderedDict: eval result, key is the eval metric and value in the corresponding metric value.
        """
        if not eval_data:
            return

        if load_best_model:
            checkpoint_file = model_file or self.saved_model_file
            print("debug:", checkpoint_file)
            checkpoint = torch.load(checkpoint_file, map_location=torch.device("cpu"))
            self.model.load_state_dict(checkpoint['state_dict'])
            self.model.load_other_parameter(checkpoint.get('other_parameter'))
            message_output = 'Loading model structure and parameters from {}'.format(checkpoint_file)
            self.logger.info(message_output)

        self.model.eval()

        if isinstance(eval_data, FullSortEvalDataLoader):
            # this way
            eval_func = self._full_sort_batch_eval
            if self.item_tensor is None:
                self.item_tensor = eval_data.dataset.get_item_feature().to(self.device)
        else:
            eval_func = self._neg_sample_batch_eval
        if self.config['eval_type'] == EvaluatorType.RANKING:
            self.tot_item_num = eval_data.dataset.item_num

        iter_data = (
            tqdm(
                eval_data,
                total=len(eval_data),
                ncols=100,
                desc=set_color(f"Evaluate   ", 'pink'),
            ) if show_progress else eval_data
        )
        num_sample = 0
        for batch_idx, batched_data in enumerate(iter_data):
            num_sample += len(batched_data)
            interaction, scores, positive_u, positive_i = eval_func(batched_data)
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(set_color('GPU RAM: ' + get_gpu_usage(self.device), 'yellow'))
            self.eval_collector.eval_batch_collect(scores, interaction, positive_u, positive_i)
        self.eval_collector.model_collect(self.model)
        struct = self.eval_collector.get_data_struct()
        result = self.evaluator.evaluate(struct)

        # combine results from multiple ranks
        result = self._map_reduce(result, num_sample)

        self.wandblogger.log_eval_metrics(result, head='eval')

        return result

    def _map_reduce(self, result, num_sample):
        gather_result = {}
        total_sample = [
            torch.zeros(1).to(self.device) for _ in range(self.config["world_size"])
        ]
        torch.distributed.all_gather(
            total_sample, torch.Tensor([num_sample]).to(self.device)
        )
        total_sample = torch.cat(total_sample, 0)
        total_sample = torch.sum(total_sample).item()
        for key, value in result.items():
            result[key] = torch.Tensor([value * num_sample]).to(self.device)
            gather_result[key] = [
                torch.zeros_like(result[key]).to(self.device)
                for _ in range(self.config["world_size"])
            ]
            torch.distributed.all_gather(gather_result[key], result[key])
            gather_result[key] = torch.cat(gather_result[key], dim=0)
            gather_result[key] = round(
                torch.sum(gather_result[key]).item() / total_sample,
                self.config["metric_decimal_place"],
                )
        return gather_result

    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
        r"""Train the model based on the train data and the valid data.

        Args:
            train_data (DataLoader): the train data
            valid_data (DataLoader, optional): the valid data, default: None.
                                               If it's None, the early_stopping is invalid.
            verbose (bool, optional): whether to write training and evaluation information to logger, default: True
            saved (bool, optional): whether to save the model parameters, default: True
            show_progress (bool): Show the progress of training epoch and evaluate epoch. Defaults to ``False``.
            callback_fn (callable): Optional callback function executed at end of epoch.
                                    Includes (epoch_idx, valid_score) input arguments.

        Returns:
             (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """
        if saved and self.start_epoch >= self.epochs and self.lrank == 0:
            self._save_checkpoint(-1, verbose=verbose)

        self.eval_collector.data_collect(train_data)
        if self.config['train_neg_sample_args'].get('dynamic', 'none') != 'none':
            train_data.get_model(self.model)
        valid_step = 0

        for epoch_idx in range(self.start_epoch, self.epochs):
            # train
            training_start_time = time()
            train_loss = self._train_epoch(train_data, epoch_idx, show_progress=show_progress)

            dist.barrier()
            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
            if verbose:
                self.logger.info(train_loss_output)
            if self.lrank == 0:
                self.wandblogger.log_metrics({'epoch': epoch_idx, 'train_loss': train_loss, 'train_step':epoch_idx}, head='train')

            dist.barrier()
            # eval
            if self.eval_step <= 0 or not valid_data:
                if saved and self.lrank == 0:
                    self._save_checkpoint(epoch_idx, verbose=verbose)
                continue
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(valid_data, show_progress=show_progress)
                self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                    valid_score,
                    self.best_valid_score,
                    self.cur_step,
                    max_step=self.stopping_step,
                    bigger=self.valid_metric_bigger
                )
                valid_end_time = time()
                valid_score_output = (set_color("epoch %d evaluating", 'green') + " [" + set_color("time", 'blue')
                                      + ": %.2fs, " + set_color("valid_score", 'blue') + ": %f]") % \
                                     (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = set_color('valid result', 'blue') + ': \n' + dict2str(valid_result)
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                if self.rank == 0:
                    self.tensorboard.add_scalar('Vaild_score', valid_score, epoch_idx)
                self.wandblogger.log_metrics({**valid_result, 'valid_step': valid_step}, head='valid')

                if update_flag:
                    if saved and self.lrank == 0:
                        self._save_checkpoint(epoch_idx, verbose=verbose)
                    self.best_valid_result = valid_result

                if callback_fn:
                    callback_fn(epoch_idx, valid_score)

                if stop_flag:
                    stop_output = 'Finished training, best eval result in epoch %d' % \
                                  (epoch_idx - self.cur_step * self.eval_step)
                    if verbose:
                        self.logger.info(stop_output)
                    break

                valid_step+=1

        dist.barrier()
        return self.best_valid_score, self.best_valid_result


class DDPMISSRecTrainer(DDPTrainer):
    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        r"""Train the model in an epoch

        Args:
            train_data (DataLoader): The train data.
            epoch_idx (int): The current epoch id.
            loss_func (function): The loss function of :attr:`model`. If it is ``None``, the loss function will be
                :attr:`self.model.calculate_loss`. Defaults to ``None``.
            show_progress (bool): Show the progress of training epoch. Defaults to ``False``.

        Returns:
            float/tuple: The sum of loss returned by all batches in this epoch. If the loss in each batch contains
            multiple parts and the model return these multiple parts loss instead of the sum of loss, it will return a
            tuple which includes the sum of loss in each part.
        """
        self._multimodal_interest_discovery(epoch_idx)
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        iter_data = (
            tqdm(
                train_data,
                total=len(train_data),
                ncols=100,
                desc=set_color(f"Train {epoch_idx:>5}", 'pink'),
            ) if show_progress else train_data
        )
        for batch_idx, interaction in enumerate(iter_data):
            interaction = interaction.to(self.device)
            interaction = self._trans_dataload(interaction)
            self.optimizer.zero_grad()
            losses = loss_func(interaction)

            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
            self._check_nan(loss)
            loss.backward()
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            self.optimizer.step()
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(set_color('GPU RAM: ' + get_gpu_usage(self.device), 'yellow'))
        return total_loss

    @torch.no_grad()
    def _multimodal_interest_discovery(self, epoch_idx=None):
        self.logger.info(f'Discovering multi-modal user interest' + ('' if epoch_idx is None else f' before {epoch_idx}-th epoch'))
        text_embedding_num, img_embedding_num = 0, 0
        if 'text' in self.model.modal_type:
            text_embedding_num = self.model.plm_embedding.num_embeddings - 1
        if 'img' in self.model.modal_type:
            img_embedding_num = self.model.img_embedding.num_embeddings - 1
        
        if self.rank == 0:
            all_embedding_list = []
            self.model.eval()
            # get modality embeddings
            if 'text' in self.model.modal_type:
                all_text_embeddings = self.model.text_adaptor(self.model.plm_embedding.weight.data[1:])
                all_embedding_list.append(all_text_embeddings)
            if 'img' in self.model.modal_type:
                all_img_embeddings = self.model.img_adaptor(self.model.img_embedding.weight.data[1:])
                all_embedding_list.append(all_img_embeddings)
            all_embedding = torch.cat(all_embedding_list, dim=0)
            
            # multi-modal interest discovery
            # cluster_idx, centroids = cluster_dpc_knn(
            cluster_idx, centroids = cluster_kmeans(
                all_embedding, 
                cluster_num=self.model.num_interest, 
                # local_size=self.model.knn_local_size
            )
            cluster_idx += 1 # offset 1 for padding_idx
            # all_reduce requires all tensors contigous
            cluster_idx = cluster_idx.contiguous()
            centroids = centroids.contiguous()
            del all_embedding_list, all_embedding
        else: # rank != 0
            cluster_idx_num = text_embedding_num + img_embedding_num
            cluster_idx = torch.zeros(cluster_idx_num, dtype=torch.long, device=self.model.device)
            centroids = torch.zeros_like(self.model.interest_embeddings[1:])
        
        dist.all_reduce(cluster_idx)
        dist.all_reduce(centroids)
        
        # write back to the model
        self.model.interest_embeddings[1:] = centroids
        read_ptr = 0
        if 'text' in self.model.modal_type:
            self.model.plm_interest_lookup_table[1: 1+text_embedding_num] = cluster_idx[read_ptr: read_ptr+text_embedding_num]
            read_ptr += text_embedding_num
        if 'img' in self.model.modal_type:
            self.model.img_interest_lookup_table[1: 1+img_embedding_num] = cluster_idx[read_ptr: read_ptr+img_embedding_num]
            read_ptr += img_embedding_num
        assert read_ptr == len(cluster_idx), f"read_ptr={read_ptr}, len(cluster_idx)={len(cluster_idx)}"
        del cluster_idx, centroids
        self.logger.info(f'Finish multi-modal interest discovery' + ('' if epoch_idx is None else f' before {epoch_idx}-th epoch'))


class DDPMISSRecPretrainTrainer(DDPPretrainTrainer):
    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        r"""Train the model in an epoch

        Args:
            train_data (DataLoader): The train data.
            epoch_idx (int): The current epoch id.
            loss_func (function): The loss function of :attr:`model`. If it is ``None``, the loss function will be
                :attr:`self.model.calculate_loss`. Defaults to ``None``.
            show_progress (bool): Show the progress of training epoch. Defaults to ``False``.

        Returns:
            float/tuple: The sum of loss returned by all batches in this epoch. If the loss in each batch contains
            multiple parts and the model return these multiple parts loss instead of the sum of loss, it will return a
            tuple which includes the sum of loss in each part.
        """
        self._multimodal_interest_discovery(train_data, epoch_idx, preproc_bs=train_data.batch_size)
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        iter_data = (
            tqdm(
                train_data,
                total=len(train_data),
                ncols=100,
                desc=set_color(f"Train {epoch_idx:>5}", 'pink'),
            ) if show_progress else train_data
        )
        for batch_idx, interaction in enumerate(iter_data):
            interaction = interaction.to(self.device)
            interaction = self._trans_dataload(interaction)
            self.optimizer.zero_grad()
            losses = loss_func(interaction)

            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
            self._check_nan(loss)
            loss.backward()
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            self.optimizer.step()
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(set_color('GPU RAM: ' + get_gpu_usage(self.device), 'yellow'))
        return total_loss

    @torch.no_grad()
    def _multimodal_interest_discovery(self, dataloader, epoch_idx=None, preproc_bs=512):
        self.logger.info(f'Discovering multi-modal user interest' + ('' if epoch_idx is None else f' before {epoch_idx}-th epoch'))
        text_embedding_num, img_embedding_num = 0, 0
        if 'text' in self.model.modal_type:
            text_embedding_num = dataloader.dataset.plm_embedding.num_embeddings - 1
        if 'img' in self.model.modal_type:
            img_embedding_num = dataloader.dataset.img_embedding.num_embeddings - 1
        
        if self.rank == 0:
            all_embedding_list = []
            self.model.eval()
            # get modality embeddings
            if 'text' in self.model.modal_type:
                all_text_embeddings = []
                plm_embedding_data = dataloader.dataset.plm_embedding.weight.data
                for ptr in range(1, len(plm_embedding_data), preproc_bs):
                    batch_data = plm_embedding_data[ptr: ptr + preproc_bs].to(self.model.device)
                    text_embeddings = self.model.text_adaptor(batch_data)
                    del batch_data
                    all_text_embeddings.append(text_embeddings.to(plm_embedding_data.device))
                    del text_embeddings
                all_text_embeddings = torch.cat(all_text_embeddings, dim=0)
                all_embedding_list.append(all_text_embeddings)
            if 'img' in self.model.modal_type:
                all_img_embeddings = []
                img_embedding_data = dataloader.dataset.img_embedding.weight.data
                for ptr in range(1, len(img_embedding_data), preproc_bs):
                    batch_data = img_embedding_data[ptr: ptr + preproc_bs].to(self.model.device)
                    img_embeddings = self.model.img_adaptor(batch_data)
                    del batch_data
                    all_img_embeddings.append(img_embeddings.to(img_embedding_data.device))
                    del img_embeddings
                all_img_embeddings = torch.cat(all_img_embeddings, dim=0)
                all_embedding_list.append(all_img_embeddings)
            all_embedding = torch.cat(all_embedding_list, dim=0)
            
            # multi-modal interest discovery
            # cluster_idx, centroids = cluster_dpc_knn(
            cluster_idx, centroids = cluster_kmeans(
                all_embedding, 
                cluster_num=dataloader.dataset.num_interest, 
                # local_size=dataloader.dataset.knn_local_size
            )
            cluster_idx += 1 # offset 1 for padding_idx
            # all_reduce requires all tensors contigous
            cluster_idx = cluster_idx.contiguous()
            centroids = centroids.contiguous()
            kmeans_device = cluster_idx.device
            del all_embedding_list, all_embedding
        else: # rank != 0
            cluster_idx_num = text_embedding_num + img_embedding_num
            cluster_idx = torch.zeros(cluster_idx_num, dtype=torch.long, device=dataloader.dataset.interest_embeddings.device)
            centroids = torch.zeros_like(dataloader.dataset.interest_embeddings[1:])
            kmeans_device = cluster_idx.device
        
        cluster_idx = cluster_idx.to(self.model.device)
        centroids = centroids.to(self.model.device)
        dist.all_reduce(cluster_idx)
        dist.all_reduce(centroids)
        cluster_idx = cluster_idx.to(kmeans_device)
        centroids = centroids.to(kmeans_device)
        
        # write back to the model
        dataloader.dataset.interest_embeddings[1:] = centroids
        read_ptr = 0
        if 'text' in self.model.modal_type:
            if self.model.train_stage == 'pretrain':
                dataloader.dataset.plm_interest_lookup_table[1: 1+text_embedding_num] = cluster_idx[read_ptr: read_ptr+text_embedding_num]
            else:
                dataloader.dataset.plm_interest_lookup_table[1: 1+text_embedding_num] = cluster_idx[read_ptr: read_ptr+text_embedding_num]
            read_ptr += text_embedding_num
        if 'img' in self.model.modal_type:
            dataloader.dataset.img_interest_lookup_table[1: 1+img_embedding_num] = cluster_idx[read_ptr: read_ptr+img_embedding_num]
            read_ptr += img_embedding_num
        assert read_ptr == len(cluster_idx), f"read_ptr={read_ptr}, len(cluster_idx)={len(cluster_idx)}"
        del cluster_idx, centroids
        self.logger.info(f'Finish multi-modal interest discovery' + ('' if epoch_idx is None else f' before {epoch_idx}-th epoch'))
