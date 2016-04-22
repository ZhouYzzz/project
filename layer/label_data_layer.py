from data_layer import DataLayer
from database import CUHK03
from utils import CHECK
import numpy as np
from time import time

idxrg = np.arange(1945) # 200 classes
BATCH = 256

class LabelDataLayer(DataLayer):
    def check(self, bottom, top):
        CHECK.EQ(len(top), 2)
        pass

    def batch_advancer(self):
        batch = np.random.choice(idxrg, BATCH)
        data = list()
        for idx in batch:
            data.append(self.db.getd(idx))

        data = np.vstack(data)
        label = self.db.getl(batch)
        self.buffer = (data, label)

    def init(self):
        self.db = CUHK03()
        self.db.load()
        pass
