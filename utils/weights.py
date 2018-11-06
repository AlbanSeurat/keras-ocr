
class WeightsDumper:

    def __init__(self, model, name = "model"):
        self.path = "data/%s.h5" % name
        self.model = model
        self.counter = 0

    def dump(self):
        if self.counter % 50 == 0:
            self.model.save(self.path)
        self.counter += 1

    def restore(self):
        try:
            self.model.load_weights(self.path)
        except OSError:
            print("weights are not loaded\n")
            pass
