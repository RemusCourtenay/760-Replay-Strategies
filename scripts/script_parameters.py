class ScriptParameters:

    def __init__(self, num_memories, epochs,
                 num_filters_1=4, kernel_size_1=(5, 5), input_shape=(28, 28, 1),
                 max_pooling_shape=(2, 2),
                 num_filters_2=8, kernel_size_2=(3, 3),
                 activation_type='relu',
                 dense_layer_size=10):
        self.num_memories = num_memories
        self.epochs = epochs
        self.num_filters_1 = num_filters_1
        self.kernel_size_1 = kernel_size_1
        self.input_shape = input_shape
        self.max_pooling_shape = max_pooling_shape
        self.num_filters_2 = num_filters_2
        self.kernel_size_2 = kernel_size_2
        self.activation_type = activation_type
        self.dense_layer_size = dense_layer_size
