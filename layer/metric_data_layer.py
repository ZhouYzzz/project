from data_layer import DataLayer
from cuhk03 import CUHK03

class MetricDataLayer(DataLayer):
    def batch_advancer(self):
        pass

    def init(self):
        self.db = CUHK03()
        pass