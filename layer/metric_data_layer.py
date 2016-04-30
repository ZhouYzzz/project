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
        (data, label) = self.db.gen_batch(BATCH)
        self.buffer = (np.vstack(data), np.vstack(label))

    def batch_forward(self, top):
        (data, label) = self.buffer
        top[0].data[...] = data
        top[1].data[...] = label

    def init(self):
        self.db = CUHK03(12661)
        self.db.load()
        pass
