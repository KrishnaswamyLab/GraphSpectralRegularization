import tensorflow as tf

class TestHook(tf.train.SessionRunHook):
    def begin(self):
        print([(var.name, var.shape) for var in tf.trainable_variables()])

class EarlyStoppingHook(tf.train.SessionRunHook):
    def __init__(self, monitor='mse_loss', thresh = 0.):
        print('Early stopping mse thresh =', thresh)
        self.monitor = monitor
        self.thresh = thresh
    def begin(self):
        graph = tf.get_default_graph()
        #print([n.name for n in graph.as_graph_def().node])
        self.monitor = graph.as_graph_element(self.monitor)
        if isinstance(self.monitor, tf.Operation):
            print('ASASDFASDF is Operation')
            self.monitor = self.monitor.outputs[0]
        print(self.monitor)
    def before_run(self, run_context):
        return tf.train.SessionRunArgs(self.monitor)
    def after_run(self, run_context, run_values):
        current = run_values.results
        print(type(current))
        print(current)
        if current < self.thresh:
            print('Requesting early stopping with loss=', current)
            run_context.request_stop()
