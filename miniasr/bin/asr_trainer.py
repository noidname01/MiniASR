'''
    File      [ asr_trainer.py ]
    Author    [ Heng-Jui Chang (NTUEE) ]
    Synopsis  [ Creates ASR trainer. ]
'''

import logging
import pytorch_lightning as pl
import requests
#from pytorch_lightning.loggers import CSVLogger

from miniasr.data.dataloader import create_dataloader
from miniasr.utils import load_from_checkpoint


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

        # Logger
        #csv_logger = CSVLogger("./logs", name="mini_asr_training_log")

        class MyPrintingCallback(pl.Callback):

          
            def on_validation_end(self, trainer, pl_module):
                if trainer.sanity_checking:
                    return
                print("epoch: " + str(trainer.current_epoch))
                logging.info("VAL_CER: " + str(trainer.callback_metrics['val_cer'].item()))
                logging.info("VAL_WER: " + str(trainer.callback_metrics['val_wer'].item()))
                logging.info("VAL_LOSS: " + str(trainer.callback_metrics['val_loss'].item()))
                logging.info("TRAIN_LOSS: " + str(trainer.callback_metrics['train_loss'].item()))

                
                data = {
                    "epoch": str(trainer.current_epoch),
                    "val_cer": str(trainer.callback_metrics['val_cer'].item()),
                    "val_wer": str(trainer.callback_metrics['val_wer'].item()),
                    "val_loss": str(trainer.callback_metrics['val_loss'].item()),
                    "train_loss": str(trainer.callback_metrics['train_loss'].item())
                }

                req = requests.post("https://online-logs-viewer.herokuapp.com/objects", data=data)
                print(req)

                logging.info('\n\nValidation loop ends.\n\n')
                
                
            
            def on_validation_start(self, trainer, pl_module):
                if trainer.sanity_checking:
                    return
                logging.info('\n\nValidation loop starts.\n\n')
                

        custom_callback = MyPrintingCallback()


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
            callbacks=[checkpoint_callback],
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
