import time

import torch
import torch.nn.functional as functional


class Trainer(object):
    def __init__(self, optimizer, scheduler, device):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

    def train(self, dataloader, model, res, fixed=False):
        model.train()
        sign_loss_meter, loss_meter = 0, 0

        start_time = time.time()
        for i, (data, target) in enumerate(dataloader):
            data = data.to(self.device)
            target = target.to(self.device)

            self.optimizer.zero_grad()

            pred = model(data)
            loss = torch.tensor(0.).to(self.device) if fixed else functional.cross_entropy(pred, target)
            sign_loss = torch.tensor(0.).to(self.device)

            # add up sign loss
            if res is not None:
                sign_loss = res(model).to(self.device)

            (loss + sign_loss).backward()
            self.optimizer.step()

            sign_loss_meter += sign_loss.item()
            loss_meter += loss.item()

        if self.scheduler is not None:
            self.scheduler.step()

        return {'loss': loss_meter / len(dataloader),
                'sign_loss': sign_loss_meter / len(dataloader),
                'time': time.time() - start_time}

    def test(self, dataloader, model, res):
        model.eval()
        loss_meter, acc_meter, count, sign_acc = 0, 0, 0, 0

        start_time = time.time()
        with torch.no_grad():
            for load in dataloader:
                data, target = load[:2]
                data = data.to(self.device)
                target = target.to(self.device)

                pred = model(data)
                loss_meter += functional.cross_entropy(pred, target, reduction='sum').item()
                pred = pred.max(1, keepdim=True)[1]

                acc_meter += pred.eq(target.view_as(pred)).sum().item()
                count += data.size(0)

            if res is not None:
                sign_acc = res(model, accuracy=True)

        return {'loss': loss_meter / count,
                'acc': 100 * acc_meter / count,
                'sign acc': sign_acc,
                'time': time.time() - start_time}
