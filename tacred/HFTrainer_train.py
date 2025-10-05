import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from datasets import load_dataset, Image
from transformers import DefaultDataCollator, TrainingArguments, Trainer
from huggingface_hub import PyTorchModelHubMixin
import evaluate
import numpy as np


class BasicNet(nn.Module,PyTorchModelHubMixin,tags=["image-classification"]):
    def __init__(self,channels):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.conv1 = nn.Conv2d(channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, image,labels=None):
        # the labels parameter allows us to finetune our model
        # with the Trainer API easily
        x = F.relu(F.max_pool2d(self.conv1(image), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        logits = F.log_softmax(x,dim=-1)
        if labels != None :
            # this will make your AI compatible with the trainer API
            loss = self.criterion(logits, labels)
            return {"loss": loss, "logits": logits}
        return logits
    

def collate_fn(examples):
    images = []
    labels = []
    for example in examples:
        images.append((example["image"]))
        labels.append(example["labels"])

    images = torch.stack(images)
    labels = torch.tensor(labels)
    return {"image": images, "labels": labels}


def mnist_datasets():
    dataset = load_dataset("mnist")
    dataset = dataset.rename_column("label","labels")
    # convert to Pytorch tensors
    transform = transforms.Compose([transforms.ToTensor()])
    def to_pt(batch):
        batch["image"] = [transform(image.convert("RGB")) for image in batch["image"]]
        return batch
    train = dataset["train"].with_transform(to_pt)
    test = dataset["test"].with_transform(to_pt)

    return train, test



def entrainement():
    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)
    
    modele = BasicNet(channels=3)
    dsTRAIN, dsTEST = mnist_datasets()

    #with torch.no_grad():
    #    bt0 = collate_fn([dsTEST[0], dsTEST[1]])
    #    resu = modele(**bt0)


    training_args = TrainingArguments(
        output_dir="mon_modèle_mnist",
        logging_dir="logs_de_mon_modèle_mnist",
        # remove_unused_columns=False,
        report_to="tensorboard",   # Sinon, ça tente un rapport dans wandb.
        evaluation_strategy="steps",
        save_strategy="epoch",
        learning_rate=5e-4,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        num_train_epochs=2,
        logging_steps=100,
        push_to_hub=False,
    )
    
    trainer = Trainer(
        model=modele,
        args=training_args,
        train_dataset=dsTRAIN,
        eval_dataset=dsTEST,
        compute_metrics=compute_metrics,
    )

    trainer.train()
