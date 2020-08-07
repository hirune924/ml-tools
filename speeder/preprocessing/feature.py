

class FeatureFactory:

    def __init__(self, configs: dict, cv=None):
        self.run_name = configs['fe_name']
        self.data = configs.data
        self.fe = configs.fe
        self.cv = cv

    def create(self):
        print(self.data)