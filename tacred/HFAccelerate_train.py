import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from datasets import load_dataset, Image
#from transformers import DefaultDataCollator, TrainingArguments, Trainer
from huggingface_hub import PyTorchModelHubMixin
import evaluate
import numpy as np
from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam, AdamW
from transformers import get_scheduler
from accelerate import Accelerator
from tqdm.auto import tqdm


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



class MOYENNE:
    def __init__(self):
        self.N = 0
        self.m = 0.

    def cumuler(self, v):
        self.N += 1
        self.m += (v-self.m)/self.N
        return self.m

    def RaZ(self):
        self.N = 0
        return self.m


def entrainement():
    accuracy = evaluate.load("accuracy")

    model = BasicNet(channels=3)
    dsTRAIN, dsEVAL = mnist_datasets()

    loadTRAIN = DataLoader(dsTRAIN, batch_size=64, shuffle=True)
    loadEVAL = DataLoader(dsEVAL, batch_size=64, shuffle=False)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)
    
    def evaluation():
        model.eval()
        losses = [0.0] * len(loadEVAL)
        with torch.no_grad():
            for step, batch in enumerate(tqdm(loadEVAL)):
                outputs = model(batch["image"], labels=batch["labels"])
                losses[step] = accelerator.gather(outputs["loss"]).item()
        loss = torch.mean(torch.tensor(losses))
        
        return loss.item()


    
    accelerator = Accelerator(log_with="tensorboard", project_dir="proj_Accel")  #(fp16=True)
    config = {
        "num_iterations": 5,
        "learning_rate": 1e-3,
    }

    accelerator.init_trackers("exemple_1", config=config)
    device = accelerator.device
    #model.to(device)
    optimizer = AdamW(model.parameters(), lr = 5e-4, weight_decay=0.0)


    #Transformation du modèle et des trucs
    model, optimizer, loadTRAIN, loadEVAL = accelerator.prepare(
        model, optimizer, loadTRAIN, loadEVAL
    )

    num_train_epochs = 2
    num_update_steps_per_epoch = len(loadTRAIN)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch

    #evaluation()
    

    model.train()
    global_step=0
    train_loss=MOYENNE()
    for epoch in range(num_train_epochs):
        for step, batch in enumerate(tqdm(loadTRAIN)):
            optimizer.zero_grad()
            image, labels = batch["image"], batch["labels"]
            outputs = model(image, labels)
            logits = outputs["logits"]
            loss   = outputs["loss"]
            accelerator.backward(loss)
            optimizer.step()
            train_loss.cumuler(loss.item())
            if step % 100 == 0:
                accelerator.log({"numero step": step, "perte(train)": train_loss.m}, step=global_step)
                train_loss.RaZ()
            global_step += 1
            
    accelerator.end_training()
                

