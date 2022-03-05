import torch
from torch import nn


class SessionClassifier(nn.Module):
    def __init__(self, categorical_data, numerical_data):
        super(SessionClassifier, self).__init__()
        self.emb_layer = nn.Linear(categorical_data, categorical_data)
        self.act_emb = nn.Tanh()
        self.layer1 = nn.Linear(numerical_data + categorical_data, 40)
        self.act_1 = nn.LeakyReLU()
        self.d1 = nn.Dropout(0.4)
        self.layer2 = nn.Linear(40, 20)
        self.act_2 = nn.LeakyReLU()
        self.d2 = nn.Dropout(0.4)
        self.layer3 = nn.Linear(20, 1)
        self.f = nn.Sigmoid()

    def forward(self, x, cat_x):
        cat_x_embedded = self.emb_layer(cat_x)
        cat_x_embedded = self.act_emb(cat_x_embedded)
        x = torch.cat([x, cat_x_embedded], dim=0)
        activation1 = self.act_1(self.layer1(x))
        activation1 = self.d1(activation1)
        activation2 = self.act_2(self.layer2(activation1))
        activation2 = self.d2(activation2)
        output = self.layer3(activation2)
        output = self.f(output)
        return output
