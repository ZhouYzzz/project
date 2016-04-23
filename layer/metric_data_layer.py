from data_layer import DataLayer
from database import CUHK03
from utils import CHECK
import numpy as np

BATCH = 128

class MetricDataLayer(DataLayer):
    def check(self, bottom, top):
        CHECK.EQ(len(top), 2)
        pass

    def batch_advancer(self):
        data = list()
        label = list()
        for i in xrange(BATCH):
            d, l = self.db.gen_pair()
            data.append(d)
            label.append(l)

        self.buffer = (data, label)

    def batch_forward(self, top):
        (data, label) = self.buffer
        for i in xrange(BATCH):
            top[0].data = data[0]

    def init(self):
        self.db = CUHK03()
        self.db.load()
        pass