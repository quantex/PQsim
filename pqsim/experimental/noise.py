import numpy as np

class noise():
    def __init__(self, count, cardinality, shots):
        '''
        Instantiates a noise class.
        Args:
            count: Number of noise channels to implement.
            cardinality: A list of cardinalities of the kraus set defining each noise channel.
            shots: Number of shots over which to simulate the noisy channels.
        '''
        card_max = np.max(cardinality)
        self.entropy = np.random.randint(0,card_max+1,size=(shots,count))
        self.entropy = np.mod(self.entropy, cardinality)

        self.stats = {}
        for idx in range(shots):
            str_temp = ''
            for val in self.entropy[idx]:
                str_temp = str_temp + str(val)

            if str_temp in self.stats.keys():
                self.stats[str_temp] += 1
            else:
                self.stats.update({str_temp:1})