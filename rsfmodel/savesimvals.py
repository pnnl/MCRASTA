import numpy as np
import pandas as pd

class SaveSimVals(object):
    def __init__(self):
        self.musim = None
        self.chain = None
        self.count = None


    def save_data(self):

        msdf = pd.DataFrame(self.musim)
