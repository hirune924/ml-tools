

class Trainer:
    def __init__(self, configs: dict, cv):
        self.run_name = configs['exp_name']
        self.data = configs.data
        self.coldef = self.data.cols_definition
        self.fe = configs.fe
        self.cv = cv
