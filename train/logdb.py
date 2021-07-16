import csv
import json
import os

import torch


class LogDB(object):

    def __init__(self, args):
        self.args = args
        self.model = None
        self.res = None
        self.trainer = None
        self.train_loader = None
        self.test_loader = None
        self.num_classes = None

        self.buffer = []
        self.save_history_interval = 1
        self.device = torch.device('cuda')

        self.action = args['action']

        self.arch = args['arch']
        self.dataset = args['dataset']
        self.epochs = args['epochs']
        self.batch_size = args['batch_size']
        self.lr = args['lr']
        self.lr_config = json.load(open(f'settings/{args["lr_config"]}'))
        self.res_config = json.load(open(f'settings/{self.arch}.json'))
        self.pretrained_path = args['pretrained_path']

        self.norm_type = args['norm_type']

        self.embed = args['embed']

        # residuals
        self.threshold = args['threshold']
        self.lamda = args['lamda']
        self.divider = args['divider']

        self.logdir = f'logs/{self.arch}_{self.dataset}_{self.action}'
        if self.embed:
            self.logdir += f'_embed'

    def experiment_id(self):
        exps = [d for d in os.listdir(self.logdir) if os.path.isdir(os.path.join(self.logdir, d)) and d.isdigit()]
        files = set(map(int, exps))
        if len(files):
            return min(set(range(1, max(files) + 2)) - files)
        else:
            return 1

    def create_folder(self):
        os.makedirs(self.logdir, exist_ok=True)

        self.logdir = os.path.join(self.logdir, str(self.experiment_id()))

        os.makedirs(os.path.join(self.logdir, 'models'), exist_ok=True)

        json.dump(self.args, open(os.path.join(self.logdir, 'config.json'), 'w'), indent=4)

    def save_model(self, filename, model=None):
        if model is None:
            model = self.model

        torch.save(model.cpu().state_dict(), os.path.join(self.logdir, f'models/{filename}'))
        model.to(self.device)

    def flush_history(self, history_file, first):
        if len(self.buffer) != 0:
            columns = sorted(self.buffer[0].keys())
            with open(history_file, 'a') as file:
                writer = csv.writer(file, delimiter=',', quotechar="'", quoting=csv.QUOTE_MINIMAL)
                if first:
                    writer.writerow(columns)

                for data in self.buffer:
                    writer.writerow(list(map(lambda x: data[x], columns)))

            self.buffer.clear()

    def append_history(self, history_file, data, first=False):
        self.buffer.append(data)

        if len(self.buffer) >= self.save_history_interval:
            self.flush_history(history_file, first)
