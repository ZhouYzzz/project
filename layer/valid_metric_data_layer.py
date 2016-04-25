from data_layer import DataLayer
from database import VIPeR
from utils import CHECK
import numpy as np

# BATCH = 128

class ValidMetricDataLayer(DataLayer):
    def check(self, bottom, top):
        CHECK.EQ(len(top), 2)
        pass

    def batch_advancer(self):
        self.buffer = self.db.gen_valid()

    def batch_forward(self, top):
        (data, label) = self.buffer
        top[0].data[...] = data
        top[1].data[...] = label

    def init(self):
        self.db = VIPeR()
        pass
