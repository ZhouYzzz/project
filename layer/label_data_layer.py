from data_layer import DataLayer
from cuhk03 import CUHK03
from check import CHECK

class LabelDataLayer(DataLayer):
    def check(self, bottom, top):
        CHECK.EQ(len(top), 2)
        pass

    def batch_advancer(self):
        pass

    def init(self):
        self.db = CUHK03()
        self.db.load()
        pass