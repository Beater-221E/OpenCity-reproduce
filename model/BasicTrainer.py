import os
import sys
import time
import copy

import numpy as np
import torch
from tqdm import tqdm

from lib.data_process import get_key_from_value
from lib.logger import get_logger
from lib.metrics import All_Metrics


def _progress(it, total=None, desc="", leave=True, unit="batch"):
    """tqdm wrapper: always on for eval/lora/pretrain when OPENCITY_TQDM=1 or stderr is a TTY."""
    force = os.environ.get("OPENCITY_TQDM", "").strip() in ("1", "true", "yes")
    if not force and not (hasattr(sys.stderr, "isatty") and sys.stderr.isatty()):
        return it
    return tqdm(
        it,
        total=total,
        desc=desc,
        leave=leave,
        unit=unit,
        file=sys.stderr,
        dynamic_ncols=True,
        mininterval=0.5,
    )


class Trainer(object):
    def __init__(self, model, loss, optimizer, train_dataloader, val_dataloader, test_dataloader, scaler_dict,
                 args, scheduler):
        super(Trainer, self).__init__()
        self.model = model
        self.args = args
        self.loss = loss
        self.optimizer = optimizer
        self.num_nodes_dict = args.num_nodes_dict
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.scaler_dict = scaler_dict
        self.scheduler = scheduler
        self.batch_seen = 0
        self.best_path = os.path.join(self.args.log_dir, self.args.save_pretrain_path)
        self.loss_figure_path = os.path.join(self.args.log_dir, 'loss.png')
        # log
        if os.path.isdir(args.log_dir) == False and not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, name=args.model, debug=args.debug)
        self.logger.info('Experiment log path in: {}'.format(args.log_dir))

    def multi_train(self):
        best_model = None
        best_loss = float('inf')
        not_improved_count = 0
        # train_loss_list = []
        val_loss_list = []

        mode = getattr(self.args, "mode", "train")
        epoch_iter = _progress(
            range(self.args.epochs),
            total=self.args.epochs,
            desc=f"{mode} epochs",
            leave=True,
            unit="epoch",
        )
        for epoch in epoch_iter:
            train_epoch_loss = self.multi_train_eps(epoch)
            if hasattr(epoch_iter, "set_postfix"):
                epoch_iter.set_postfix(train_loss=f"{train_epoch_loss:.4f}")

            if train_epoch_loss > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break

            if self.args.mode != 'pretrain' and self.args.val_ratio > 0:
                # Val
                val_epoch_loss = self.multi_val_epoch(epoch)
                # Best state and early stop epoch
                val_loss_list.append(val_epoch_loss)
                if val_epoch_loss < best_loss:
                    best_loss = val_epoch_loss
                    not_improved_count = 0
                    best_state = True
                else:
                    not_improved_count += 1
                    best_state = False

                # early stop
                if self.args.early_stop:
                    if not_improved_count == self.args.early_stop_patience:
                        self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                         "Training stops.".format(self.args.early_stop_patience))
                        break

                # save the best state
                if best_state == True:
                    self.logger.info('*********************************Current best model saved!')
                    # self.test(self.model, self.args, self.scaler_dict, self.test_dataloader, self.logger)
                    best_model = copy.deepcopy(self.model.state_dict())
        # test
        if self.args.mode != 'pretrain' and self.args.val_ratio > 0:
            self.model.load_state_dict(best_model)
            self.test(self.model, self.args, self.scaler_dict, self.test_dataloader, self.logger)
        print("Pre-train finish.")


    def multi_train_eps(self, epoch=0):
        self.model.train()
        total_loss = 0
        step = 0
        n_batches = len(self.train_dataloader)
        batch_iter = _progress(
            self.train_dataloader,
            total=n_batches,
            desc=f"train ep{epoch + 1}",
            leave=False,
            unit="batch",
        )
        for inputs, targets in batch_iter:
            inputs, targets = inputs.squeeze(0).to(self.args.device), targets.squeeze(0).to(self.args.device)
            select_dataset = get_key_from_value(self.num_nodes_dict, inputs.shape[2])
            out = self.model(inputs, targets, select_dataset, batch_seen=None)
            self.optimizer.zero_grad()
            loss_pred = self.loss(out, targets[..., :self.args.output_dim], self.scaler_dict[select_dataset])
            loss = loss_pred
            loss.backward()

            # add max grad clipping
            if self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()

            # learning rate decay
            if self.args.lr_decay:
                self.scheduler.step()
            total_loss += loss.item()
            step += 1
            if hasattr(batch_iter, "set_postfix"):
                batch_iter.set_postfix(loss=f"{loss.item():.4f}", avg=f"{total_loss / step:.4f}")
            if step % self.args.log_step == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                self.logger.info(
                    "step:  " + str(step) + "  train loss is:  " + str(total_loss / step) + "  current_lr is:  " + str(
                        current_lr))
            if step % self.args.save_step == 0 and self.args.debug:
                best_model = copy.deepcopy(self.model.state_dict())
                torch.save(best_model, self.best_path)
                self.logger.info("Saving current best model to " + self.best_path)
        train_loss = total_loss / len(self.train_dataloader)
        return train_loss

    def multi_val_epoch(self, epoch):
        self.model.eval()
        total_val_loss = 0
        with torch.no_grad():
            val_iter = _progress(
                self.val_dataloader,
                total=len(self.val_dataloader),
                desc=f"val ep{epoch + 1}",
                leave=False,
                unit="batch",
            )
            for inputs, targets in val_iter:
                inputs, targets = inputs.squeeze(0).to(self.args.device), targets.squeeze(0).to(self.args.device)
                select_dataset = get_key_from_value(self.num_nodes_dict, inputs.shape[2])
                out = self.model(inputs, targets, select_dataset, batch_seen=None)
                loss_pred = self.loss(out, targets[..., 0:self.args.output_dim], self.scaler_dict[select_dataset])
                if not torch.isnan(loss_pred):
                    total_val_loss += loss_pred.item()
                val_loss = total_val_loss / len(self.val_dataloader)
        self.logger.info('**********Val Epoch {}: average Loss: {:.6f}'.format(epoch, val_loss))
        return val_loss

    @staticmethod
    def test(model, args, scaler_dict, test_dataloader, logger, path=None):
        if path != None:
            if torch.cuda.device_count() > 1:
                model.load_state_dict(torch.load(path))
            else:
                model_weights = {k.replace('module.', ''): v for k, v in torch.load(path).items()}
                model.load_state_dict(model_weights)
            model.to(args.device)
        model.eval()

        with torch.no_grad():
            mae = 0
            rmse = 0
            mape = 0
            total_count = 0
            total_mape_count = 0
            total_batch = 0
            test_iter = _progress(
                test_dataloader,
                total=len(test_dataloader),
                desc="test",
                leave=True,
                unit="batch",
            )
            for inputs, targets in test_iter:
                inputs, targets = inputs.squeeze(0).to(args.device), targets.squeeze(0).to(args.device)
                select_dataset = get_key_from_value(args.num_nodes_dict, inputs.shape[2])
                output = model(inputs, targets, select_dataset, batch_seen=None)
                if args.real_value == False:
                    output = scaler_dict[select_dataset].inverse_transform(output)
                    y_lbl = scaler_dict[select_dataset].inverse_transform(targets[..., :args.output_dim])
                else:
                    y_lbl = targets[..., :args.output_dim]
                batch_mae, batch_rmse, batch_mape, batch_mse, corr, mae_count, rmse_count, mse_count, mape_count = \
                    All_Metrics(output, y_lbl, args.mae_thresh, args.mape_thresh)
                mae += batch_mae * mae_count
                rmse += batch_mse * rmse_count
                mape += batch_mape * mape_count
                total_count += mae_count
                total_mape_count += mape_count
                total_batch += len(y_lbl)
                if args.model == 'OpenCity':
                    print(total_batch, batch_mae, batch_rmse, batch_mape, total_count, total_mape_count)
        mae /= total_count
        rmse = (rmse / total_count) ** 0.5
        mape /= total_mape_count
        print('last batch', output.shape, y_lbl.shape)
        print(total_batch, total_count, total_mape_count)

        logger.info("Average Horizon, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%, CORR:{:.4f}".format(
            mae, rmse, mape * 100, corr))
