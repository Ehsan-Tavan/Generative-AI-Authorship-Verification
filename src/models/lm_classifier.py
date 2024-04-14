"""
    Generative_AI_Authorship_Verification Project:
        models:
            lm_classifier.py

"""

# ============================ Third Party libs ============================
from typing import Type, Dict, List, Optional

import pytorch_lightning as pl
import torch
import torchmetrics
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from src.dataset import DataModule
# ============================ My packages ============================
from .pooling_model import Pooling


class LmClassifier(pl.LightningModule):
    def __init__(self,
                 model_path: str,
                 args,
                 loss_fct,
                 train_data: list,
                 dev_data: list,
                 dataset_obj,
                 num_classes: int = 2,
                 pooling_methods: List[str] = None,
                 optimizer_class: Type[torch.optim.Optimizer] = torch.optim.AdamW,
                 optimizer_params: Dict[str, object] = None,
                 ):
        super().__init__()
        self.args = args

        self.loss_fct = loss_fct

        pooling_methods = pooling_methods or ["cls"]
        optimizer_params = optimizer_params or {"lr": self.args.lr}

        # Initialization of model components
        self.model = AutoModel.from_pretrained(model_path, output_hidden_states=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.linear = torch.nn.Linear(self.model.config.hidden_size, num_classes)

        self.train_data = train_data
        self.dev_data = dev_data
        if len(pooling_methods) != 1:
            raise ValueError("Using only one type of pooling methods. ['mean', 'max', 'cls']")
        self.pooling_methods = pooling_methods
        self.pooling_model = Pooling()
        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params

        self.dataset_obj = dataset_obj

        self.trainer = None
        self.data_module = None
        self.checkpoint_callback = None

        self.accuracy = torchmetrics.Accuracy(task="binary")
        self.f_score = torchmetrics.F1Score(task="binary", average="none", num_classes=num_classes)
        self.f_score_weighted = torchmetrics.F1Score(task="binary", average="weighted",
                                                     num_classes=num_classes)
        self.f_score_macro = torchmetrics.F1Score(task="binary", average="macro",
                                                  num_classes=num_classes)

        self.save_hyperparameters()

    def predict(self,
                data: List,
                device: str = "cpu"):
        self.args.device = device
        dataset = self.dataset_obj(
            data=data,
            tokenizer=self.tokenizer,
            max_len=self.args.max_length)
        dataloader = DataLoader(dataset=dataset, batch_size=64, shuffle=False)
        predictions = []
        with torch.no_grad():
            for batch_ndx, batch in enumerate(dataloader):
                batch.pop("targets")
                batch["input_ids"] = batch["input_ids"].to(self.args.device)
                batch["attention_mask"] = batch["attention_mask"].to(self.args.device)
                out = torch.argmax(self.forward(batch), dim=1).detach().cpu().numpy()
                predictions.extend(out)
        return predictions

    def forward(self, batch: dict) -> torch.Tensor:
        output = self.model(**batch, return_dict=True)
        output = self.pooling_model(output.last_hidden_state, batch["attention_mask"],
                                    pooling_methods=self.pooling_methods)

        logits = self.linear(output[0])
        return logits

    def training_step(self, batch: dict, _):
        targets = batch["targets"].flatten()
        batch.pop("targets")
        outputs = self.forward(batch)
        loss = self.loss_fct(outputs, targets)

        metric2value = {
            "train_loss": loss,
            "train_acc": self.accuracy(torch.argmax(outputs, dim=1), targets),
            "train_f_score_macro": self.f_score_macro(torch.argmax(outputs, dim=1), targets),
            "train_f_score_weighted": self.f_score_weighted(torch.argmax(outputs, dim=1), targets)
        }

        self.log_dict(metric2value, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs, "labels": targets}

    def validation_step(self, batch: dict, _):
        targets = batch["targets"].flatten()
        batch.pop("targets")
        outputs = self.forward(batch)
        loss = self.loss_fct(outputs, targets)

        metric2value = {
            "dev_loss": loss,
            "dev_acc": self.accuracy(torch.argmax(outputs, dim=1), targets),
            "dev_f_score_macro": self.f_score_macro(torch.argmax(outputs, dim=1), targets),
            "dev_f_score_weighted": self.f_score_weighted(torch.argmax(outputs, dim=1), targets)
        }

        self.log_dict(metric2value, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch: dict, _):
        """
        Defines a single testing step for a PyTorch Lightning module.

        Args:
            batch: A batch of test data.
            _ : Additional information (not used in this function).

        Returns:
            float: The computed loss for the test step.
        """
        targets = batch["targets"].flatten()
        batch.pop("targets")
        outputs = self.forward(batch)
        loss = self.loss_fct(outputs, targets)

        metric2value = {
            "test_loss": loss,
            "test_acc": self.accuracy(torch.argmax(outputs, dim=1), targets),
            "test_f_score_macro": self.f_score_macro(torch.argmax(outputs, dim=1), targets),
            "test_f_score_weighted": self.f_score_weighted(torch.argmax(outputs, dim=1), targets)
        }

        self.log_dict(metric2value, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

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

    def configure_optimizers(self):
        """
        Configures the optimizer for training.

        Returns:
            list: A list containing the configured optimizer.
        """
        optimizer = self.optimizer_class(self.parameters(), **self.optimizer_params)
        # Return the optimizer as a list
        return [optimizer]

    def test(self):
        """
        Runs the model's test procedure using the configured trainer and data module.

        Returns:
            None
        """
        # Execute model testing
        self.trainer.test(self, datamodule=self.data_module)

    # def save(self):
    #     """
    #     Saves the best model checkpoint and the associated language model.
    #
    #     Returns:
    #         None
    #     """
    #     # Load the best model checkpoint
    #     best_model = BiEncoder.load_from_checkpoint(self.checkpoint_callback.best_model_path)
    #
    #     # Save the best model to a .pt file
    #     torch.save(best_model,
    #                self.checkpoint_callback.dirpath.replace("checkpoints", "best_model.pt"))
    #
    #     # Save the language model's pretrained weights
    #     best_model.model.save_pretrained(
    #         self.checkpoint_callback.dirpath.replace("checkpoints", "lm"))
