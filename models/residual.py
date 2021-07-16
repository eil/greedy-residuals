import torch
import torch.nn.functional as functional
import rsa


class Residual(torch.nn.Module):
    def __init__(self, m_layers=None, objective_size=None, threshold=0.1, lamda=1, divider=10, assertion=None):
        super(Residual, self).__init__()

        # hard-code configuration
        self.copyright = b'This work is done by Peking University' if assertion is None else assertion

        self.register_buffer('sig', self.keygen())
        self.objective_size = objective_size
        self.m_layers = m_layers
        self.threshold = threshold
        self.lamda = lamda
        self.divider = divider
        self.weight_distribution = None
        if assertion is None:
            print(f'Threshold {self.threshold}, adjustable factor {self.lamda}, divider {self.divider}')

    def keygen(self):
        signature_set = bytes()
        clients = 1  # if embed else 0

        for i in range(clients):
            (pubkey, privkey) = rsa.newkeys(512)
            message = self.copyright
            signature = rsa.sign(message, privkey, 'SHA-256')
            signature_set += signature
        signature_set = rsa.compute_hash(signature_set, 'SHA-256')
        b_sig = list(bin(int(signature_set.hex(), base=16)).lstrip('0b'))  # hex -> bin
        b_sig = list(map(int, b_sig))  # bin-> int
        while len(b_sig) < 256:
            b_sig.insert(0, 0)
        sig = torch.tensor([-1 if i == 0 else i for i in b_sig], dtype=torch.float)
        return sig

    def extract_weight(self, model):
        extraction = None
        for i, (name, param) in enumerate(model.named_parameters()):
            if 'weight' in name:
                if name in self.m_layers:
                    layer_weight = param.view(-1)[
                                   :param.numel() // self.objective_size[1] * self.objective_size[1]]
                    layer_weight = functional.adaptive_avg_pool1d(layer_weight[None, None],
                                                                  self.objective_size[0] * self.objective_size[
                                                                      1]).squeeze(
                        0).view(self.objective_size)
                    if extraction is None:
                        extraction = layer_weight
                    else:
                        extraction += layer_weight

        extraction /= len(self.m_layers)
        return extraction

    def construct_residual(self, weight_extraction):
        if self.divider != 0:
            idx = torch.argsort(torch.abs(weight_extraction),
                                dim=1)[:, :int(self.objective_size[1] / self.divider + 0.5)]
            for i in range(self.objective_size[0]):
                weight_extraction[i, idx[i]] = 0

        return torch.mean(weight_extraction, dim=1)

    def forward(self, model, accuracy=False):
        weight_extraction = self.extract_weight(model)
        self.weight_distribution = weight_extraction

        pred_raw_sig = self.construct_residual(weight_extraction)
        if accuracy:
            return torch.as_tensor(torch.sign(pred_raw_sig) == self.sig).float().mean().item()
        else:
            return self.lamda * functional.relu(self.threshold - self.sig.view(-1) * pred_raw_sig.view(-1)).sum()

