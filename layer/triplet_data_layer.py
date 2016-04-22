from data_layer import DataLayer
from database import CUHK03
from check import CHECK

class TripletDataLayer(DataLayer):
    def check(self, bottom, top):
        CHECK.EQ(len(top), 2)
        pass

    def batch_advancer(self):
        data, label = self.db.get(0)
        self.buffer = (data, label)

    def init(self):
        self.db = CUHK03()
        self.db.load()
        pass
