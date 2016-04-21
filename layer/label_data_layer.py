from data_layer import DataLayer
from cuhk03 import CUHK03
from check import CHECK
import numpy as np
from time import time
# from multiprocessing import Pool

idxrg = np.arange(1945)
BATCH = 128

class LabelDataLayer(DataLayer):
    def check(self, bottom, top):
        CHECK.EQ(len(top), 2)
        pass

    def batch_advancer(self):
        #t = time()
        batch = np.random.choice(idxrg, BATCH)
        #data, label = self.db.get(np.random.choice(idxrg,BATCH,False))
        #self.buffer = (data, label)
        data = list()
        for idx in batch:
            data.append(self.db.getd(idx))

        #print 'Pool took', time()-t
        data = np.concatenate(data)
        label = self.db.getl(batch)
        self.buffer = (data, label)
        #print 'BATCH ADVANCER TOOK', time()-t

    def init(self):
        self.db = CUHK03()
        self.db.load()
        pass
