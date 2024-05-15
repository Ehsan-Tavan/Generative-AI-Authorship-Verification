"""
    Generative_AI_Authorship_Verification Project:
        models:
            bart_paraphraser.py

"""

# ============================ Third Party libs =======================
import pytorch_lightning as pl
from transformers import BartTokenizer, BartForConditionalGeneration, Adafactor
from typing import Optional
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# ============================ My packages ============================
from src.dataset import DataModule


class BartParaphraser(pl.LightningModule):
    def __init__(self, lm_model_path, learning_rate, train_data, dev_data, dataset_obj, args):
        super().__init__()
        self.model = BartForConditionalGeneration.from_pretrained(lm_model_path)
        self.tokenizer = BartTokenizer.from_pretrained(lm_model_path)
        self.learning_rate = learning_rate
        self.train_data = train_data
        self.dev_data = dev_data
        self.dataset_obj = dataset_obj
        self.args = args

        self.targets = []
        self.predictions = []

        self.data_module = None

        self.save_hyperparameters()

    def forward(self, batch):
        outputs = self.model(**batch["input_sequences"], labels=batch["target_sequences"])
        return outputs.loss

    def training_step(self, batch: dict, _: int):
        loss = self.forward(batch)
        metrics2value = {"train_loss": loss}

        self.log_dict(metrics2value, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {"loss": loss}

    def validation_step(self, batch: dict, _: int):
        loss = self.forward(batch)

        metrics2value = {"val_loss": loss}

        self.log_dict(metrics2value, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {"loss": loss}

    def test_step(self, batch: dict, _: int):
        loss = self.forward(batch)
        metrics2value = {"test_loss": loss}

        self.log_dict(metrics2value, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {"loss": loss}

    def configure_optimizers(self):
        """
        Module defines optimizer

        :return: optimizer
        """

        optimizer = Adafactor(
            self.parameters(),
            lr=self.learning_rate,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=None,
            weight_decay=0.0,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )
        return [optimizer]

    def create_trainer(self,
                       check_point_monitor: Optional[str] = None,
                       check_point_mode: Optional[str] = None,
                       early_stopping_monitor: Optional[str] = None,
                       early_stopping_patience: Optional[int] = None):
        """
        Creates and configures a PyTorch Lightning trainer for training the model.

        Args:
            check_point_monitor: The metric to monitor for model checkpointing.
            check_point_mode: The mode for the checkpointing metric (e.g., 'min' or 'max').
            early_stopping_monitor: The metric to monitor for early stopping.
            early_stopping_patience: The number of epochs with no improvement after which
                                     training will stop.

        Returns:
            pytorch_lightning.Trainer: The configured PyTorch Lightning trainer object.
        """
        callbacks = []
        # Add checkpoint callback if specified
        if check_point_monitor and check_point_mode:
            check_point_filename = "QTag-{epoch:02d}-{" + check_point_monitor + ":.2f}"
            self.checkpoint_callback = ModelCheckpoint(monitor=check_point_monitor,
                                                       filename=check_point_filename,
                                                       save_top_k=self.args.save_top_k,
                                                       mode=check_point_mode)
            callbacks.append(self.checkpoint_callback)
        # Add early stopping callback if specified
        if early_stopping_monitor and early_stopping_patience:
            early_stopping_callback = EarlyStopping(monitor=early_stopping_monitor,
                                                    patience=early_stopping_patience)
            callbacks.append(early_stopping_callback)

        logger = CSVLogger(self.args.saved_model_path, name=self.args.model_name)

        trainer = pl.Trainer(max_epochs=self.args.num_train_epochs,
                             devices=[int(self.args.device[-1])],
                             callbacks=callbacks, min_epochs=self.args.min_epochs,
                             logger=logger)

        return trainer

    def fit(self,
            check_point_monitor: Optional[str] = None,
            check_point_mode: Optional[str] = None,
            early_stopping_monitor: Optional[str] = None,
            early_stopping_patience: Optional[int] = None
            ):
        """
        Fits the model using the provided training and evaluation settings.

        Args:
            check_point_monitor: The metric to monitor for model checkpointing.
            check_point_mode: The mode for the checkpointing metric (e.g., 'min' or 'max').
            early_stopping_monitor: The metric to monitor for early stopping.
            early_stopping_patience: The number of epochs with no improvement after
                                     which training will stop.

        Returns:
            None
        """
        # Create and configure the PyTorch Lightning trainer
        self.trainer = self.create_trainer(check_point_monitor=check_point_monitor,
                                           check_point_mode=check_point_mode,
                                           early_stopping_monitor=early_stopping_monitor,
                                           early_stopping_patience=early_stopping_patience)

        # Create and configure data module for negative sampling
        self.data_module = DataModule(config=self.args,
                                      train_data=self.train_data,
                                      dev_data=self.dev_data,
                                      test_data=self.dev_data,
                                      dataset_obj=self.dataset_obj,
                                      tokenizer=self.tokenizer)

        # Start the model training using the configured trainer and data module
        self.trainer.fit(self, datamodule=self.data_module)

