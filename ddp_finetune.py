import argparse
from logging import getLogger
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from recbole.config import Config
from recbole.data import data_preparation
from recbole.utils import init_seed, init_logger, set_color

from missrec import MISSRec
from data.dataset import MISSRecDataset
from trainer import DDPMISSRecTrainer
import os


def finetune(rank, world_size, dataset, pretrained_file, mode='transductive', 
             fix_enc=True, fix_plm=False, fix_adaptor=False, **kwargs):
    # configurations initialization
    props = ['props/MISSRec.yaml', 'props/finetune.yaml']
    if rank == 0:
        print('DDP finetune on:', dataset)
        print(props)

    # configurations initialization
    print("world size", world_size, torch.cuda.device_count())
    kwargs.update({'ddp': True, 'rank': rank, 'world_size': world_size})
    config = Config(model=MISSRec, dataset=dataset, config_file_list=props, config_dict=kwargs)
    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    if config['rank'] not in [-1, 0]:
        config['state'] = 'warning'
    init_logger(config)
    logger = getLogger()
    if config['train_stage'] != mode + '_ft': # modify the training mode
        logger.info(f"Change train stage from \'{config['train_stage']}\' to \'{mode + '_ft'}\'")
        config['train_stage'] = mode + '_ft'
    logger.info(config)

    # dataset filtering
    dataset = MISSRecDataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    model = MISSRec(config, train_data.dataset)
    # count trainable parameters
    if rank == 0:
        trainable_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_params.append(name)
        logger.log(level=20, msg=f'Trainable parameters: {trainable_params}')
    
    # Load pre-trained model
    if pretrained_file != '':
        checkpoint = torch.load(pretrained_file, map_location=torch.device('cpu'))
        logger.info(f'Loading from {pretrained_file}')
        logger.info(f'Transfer [{checkpoint["config"]["dataset"]}] -> [{dataset}]')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        if fix_enc:
            logger.info(f'Fix encoder parameters.')
            for _ in model.position_embedding.parameters():
                _.requires_grad = False
            for _ in model.trm_model.encoder.parameters():
                _.requires_grad = False
        if fix_plm:
            logger.info('Fix pre-trained language model.')
            for _ in model.plm_model.parameters():
                _.requires_grad = False
        if fix_adaptor:
            logger.info('Fix adapter module.')
            for _ in model.moe_adaptor.parameters():
                _.requires_grad = False
    logger.info(model)

    # trainer loading and initialization
    trainer = DDPMISSRecTrainer(config, model)
    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=(rank == 0)
    )

    # model evaluation
    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=(rank == 0))

    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')

    dist.destroy_process_group()

    return config['model'], config['dataset'], {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='Scientific_mm_full', help='dataset name')
    parser.add_argument('-p', type=str, default='', help='pre-trained model path')
    parser.add_argument('-f', type=bool, default=True)
    parser.add_argument('-port', type=str, default='12355', help='port for ddp')
    parser.add_argument('--fix_plm', action='store_true')
    parser.add_argument('--fix_adaptor', action='store_true')
    parser.add_argument('-mode', type=str, default='transductive') # transductive (w/ ID) / inductive (w/o ID)
    args, unparsed = parser.parse_known_args()

    n_gpus = torch.cuda.device_count()
    world_size = n_gpus

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port

    mp.spawn(finetune,
             args=(world_size, args.d, args.p, args.mode, args.f, args.fix_plm, args.fix_adaptor),
             nprocs=world_size,
             join=True)
