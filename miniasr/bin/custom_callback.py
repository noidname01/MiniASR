
import logging
import pytorch_lightning as pl
import requests

class CustomCallback(pl.Callback):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args

    def on_epoch_end(self, trainer, pl_module):
        try:
            requestBody = {
                "content": "Epoch " + str(trainer.current_epoch) + " finished!",
                "title": self.args.title
            }
            req = requests.post("https://online-logs-viewer.herokuapp.com/logs", json=requestBody)
            logging.info(req)
        except Exception as e:
            logging.error("An error ocurred when posting to Online-Logs-Viewer: " + str(e))
    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        logging.info("EPOCH: " + str(trainer.current_epoch))
        logging.info("-"*20)
        logging.info("VAL_CER: " + str(trainer.callback_metrics['val_cer'].item()))
        logging.info("VAL_WER: " + str(trainer.callback_metrics['val_wer'].item()))
        logging.info("VAL_LOSS: " + str(trainer.callback_metrics['val_loss'].item()))
        logging.info("TRAIN_LOSS: " + str(trainer.callback_metrics['train_loss'].item()))
        
        try:
            requestBody = {
                "content": {
                    "epoch": trainer.current_epoch,
                    "val_cer": trainer.callback_metrics['val_cer'].item(),
                    "val_wer": trainer.callback_metrics['val_wer'].item(),
                    "val_loss": trainer.callback_metrics['val_loss'].item(),
                    "train_loss": trainer.callback_metrics['train_loss'].item()
                },
                "title": self.args.title
            }
            req = requests.post("https://online-logs-viewer.herokuapp.com/objects", json=requestBody)
            logging.info(req)
        except Exception as e:
            logging.error("An error ocurred when posting to Online-Logs-Viewer: " + str(e))
        logging.info('\n\nValidation loop ends.\n\n')
    def on_validation_start(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        logging.info('\n\nValidation loop starts.\n\n')
