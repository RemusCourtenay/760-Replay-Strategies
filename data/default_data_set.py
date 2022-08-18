from data.data_set import DataSet


# Hardcoded dataset class for use in proof of concept
class DefaultDataSet(DataSet):

    def __init__(self):
        # tasks is a list or array of sets of training data ordered into the separate tasks that they represent.
        self.tasks = None
        # current_training_data is a set of training data made up of the currently learning task data plus whichever
        # options where selected by the selection policy
        self.current_training_data = None
        # validation_data is the set of data used to check that the model isn't over-fitting.
        self.validation_data = None

    def select_training_set(self):
        # TODO...
        pass

    def get_tasks(self):
        # TODO...
        pass

    def get_training_set(self):
        # TODO...
        pass

    def get_training_labels(self):
        # TODO...
        pass

    def get_validation_set(self):
        # TODO...
        pass

    def get_validation_labels(self):
        # TODO...
        pass
