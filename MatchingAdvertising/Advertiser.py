import numpy as np


class Advertiser:
    def __init__(self, bid, publisher,budget, d_budget):
        self.bid = bid
        self.publisher = publisher
        self.budget = budget
        self.d_budget = d_budget
