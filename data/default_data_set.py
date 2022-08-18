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

    def update_training_set(self, selected_memory_data, selected_memory_labels):
        """Updates the currently selected training set by choosing a new task/tasks to train on and then adding any
        old data values that the selection policy decided to keep."""
        # TODO...
        pass

    def get_tasks(self):
        """Returns the full set of all non-validation data-label tuples as a two dimensional array sorted by task."""
        # TODO...
        pass

    def get_training_set(self):
        """Returns the set of currently selected data objects that are either in the currently training task or were
        selected by the selection policy"""
        # TODO...
        pass

    def get_training_labels(self):
        """Returns the set of currently selected data labels that are either in the currently training task or were
        selected by the selection policy"""
        # TODO...
        pass

    def get_validation_set(self):
        """Returns the set of validation data objects that are used to ensure that the model isn't over-fitting"""
        # TODO...
        pass

    def get_validation_labels(self):
        """Returns the set of validation data labels that are used to ensure that the model isn't over-fitting"""
        # TODO...
        pass
