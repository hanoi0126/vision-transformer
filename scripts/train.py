import hydra
import torch
from config import TrainConfig
from dataset import prepare_data
from libs.utils import save_checkpoint, save_experiment
from model import ViTForClassfication
from omegaconf import OmegaConf
from torch import nn, optim

import wandb


class Trainer:
    def __init__(self, model, optimizer, loss_fn, exp_name, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.exp_name = exp_name
        self.device = device

    def train(
        self, config, trainloader, testloader, epochs, save_model_every_n_epochs=0
    ):
        train_losses, test_losses, accuracies = [], [], []

        for i in range(epochs):
            train_loss = self.train_epoch(trainloader)
            accuracy, test_loss = self.evaluate(testloader)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            accuracies.append(accuracy)
            print(
                f"Epoch: {i+1}, Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}"
            )
            wandb.log(
                {
                    "epoch": i + 1,
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                    "accuracy": accuracy,
                }
            )
            if (
                save_model_every_n_epochs > 0
                and (i + 1) % save_model_every_n_epochs == 0
                and i + 1 != epochs
            ):
                print("\tSave checkpoint at epoch", i + 1)
                save_checkpoint(self.exp_name, self.model, i + 1)
        save_experiment(
            self.exp_name, config, self.model, train_losses, test_losses, accuracies
        )

    def train_epoch(self, trainloader):
        self.model.train()
        total_loss = 0
        for batch in trainloader:
            self.optimizer.zero_grad()

            batch = [t.to(self.device) for t in batch]
            images, labels = batch

            loss = self.loss_fn(self.model(images)[0], labels)
            loss.backward()

            self.optimizer.step()
            total_loss += loss.item() * len(images)

        return total_loss / len(trainloader.dataset)

    @torch.no_grad()
    def evaluate(self, testloader):
        self.model.eval()
        total_loss = 0
        correct = 0
        with torch.no_grad():
            for batch in testloader:
                batch = [t.to(self.device) for t in batch]
                images, labels = batch

                logits, _ = self.model(images)

                loss = self.loss_fn(logits, labels)
                total_loss += loss.item() * len(images)

                predictions = torch.argmax(logits, dim=1)
                correct += torch.sum(predictions == labels).item()

        accuracy = correct / len(testloader.dataset)
        avg_loss = total_loss / len(testloader.dataset)

        return accuracy, avg_loss


@hydra.main(version_base=None, config_path="../configs", config_name="train_config")
def main(config: TrainConfig) -> None:
    print(OmegaConf.to_yaml(config))
    wandb.init(
        project=config.project,
        entity=config.entity,
        name=config.datetime_dir,
        config=OmegaConf.to_container(config, resolve=True),
    )
    batch_size = config.batch_size
    epochs = config.epochs
    lr = config.lr
    device = config.device
    save_model_every_n_epochs = config.save_model_every

    trainloader, testloader, _ = prepare_data(batch_size=batch_size)

    model = ViTForClassfication(config)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss()
    trainer = Trainer(model, optimizer, loss_fn, config.exp_name, device=device)
    trainer.train(
        config,
        trainloader,
        testloader,
        epochs,
        save_model_every_n_epochs=save_model_every_n_epochs,
    )


if __name__ == "__main__":
    main()
