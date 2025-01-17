'''
    File      [ asr_trainer.py ]
    Author    [ Heng-Jui Chang (NTUEE) ]
    Synopsis  [ Creates ASR trainer. ]
'''

import logging
import pytorch_lightning as pl
from miniasr.data.dataloader import create_dataloader
from miniasr.utils import load_from_checkpoint

from miniasr.bin.custom_callback import CustomCallback


def create_asr_trainer(args, device):
    '''
        Creates ASR model and trainer. (for training)
    '''

    if args.ckpt == 'none':
        # Load data & tokenizer
        tr_loader, dv_loader, tokenizer = create_dataloader(args)

        # Create ASR model
        logging.info(f'Creating ASR model (type = {args.model.name}).')
        if args.model.name == 'ctc_asr':
            from miniasr.model.ctc_asr import ASR
        else:
            raise NotImplementedError(
                '{} ASR type is not supported.'.format(args.model.name))

        model = ASR(tokenizer, args).to(device)

        custom_callback = CustomCallback(args)
        # Create checkpoint callbacks
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=args.trainer.default_root_dir,
            **args.checkpoint_callbacks
        )

        # Create pytorch-lightning trainer
        trainer = pl.Trainer(
            accumulate_grad_batches=args.hparam.accum_grad,
            gradient_clip_val=args.hparam.grad_clip,
            callbacks=[checkpoint_callback, custom_callback],
            #logger=csv_logger,
            **args.trainer
        )
    else:
        # Load from args.ckpt (resume training)
        model, args_ckpt, tokenizer = \
            load_from_checkpoint(args.ckpt, device=device, pl_ckpt=True)
        args.model = args_ckpt.model
        if args.config == 'none':
            args.mode = args_ckpt.mode
            args.data = args_ckpt.data
            args.hparam = args_ckpt.hparam
            args.checkpoint_callbacks = args_ckpt.checkpoint_callbacks
            args.trainer = args_ckpt.trainer

        # Load data & tokenizer
        tr_loader, dv_loader, _ = create_dataloader(args)

        custom_callback = CustomCallback(args)
        # Create checkpoint callbacks
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=args.trainer.default_root_dir,
            **args.checkpoint_callbacks
        )

        # Create pytorch-lightning trainer
        trainer = pl.Trainer(
            resume_from_checkpoint=args.ckpt,
            accumulate_grad_batches=args.hparam.accum_grad,
            gradient_clip_val=args.hparam.grad_clip,
            callbacks=[checkpoint_callback, custom_callback],
            **args.trainer)

    return args, tr_loader, dv_loader, tokenizer, model, trainer


def create_asr_trainer_test(args, device):
    '''
        Creates ASR model and trainer. (for testing)
    '''

    # Load model from checkpoint
    model, args_ckpt, tokenizer = \
        load_from_checkpoint(
            args.ckpt, device=device,
            decode_args=args.decode,
            mode=args.mode)
    args.model = args_ckpt.model
    model.args = args

    # Load data & tokenizer
    _, dv_loader, _ = create_dataloader(args, tokenizer)

    # Create pytorch-lightning trainer
    trainer = pl.Trainer(**args.trainer)

    return args, None, dv_loader, tokenizer, model, trainer
