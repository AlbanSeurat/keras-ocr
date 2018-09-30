import scipy.io as sio
import numpy as np

class MatFile:

    def load(self, filename):
        data = sio.loadmat(file_name=filename, struct_as_record=False, squeeze_me=True)
        return self._check_keys(data)

    def _check_keys(self, dict):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in dict:
            if isinstance(dict[key], sio.matlab.mio5_params.mat_struct):
                dict[key] = self._todict(dict[key])
        return dict

    def _todict(self, matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        dict = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, sio.matlab.mio5_params.mat_struct):
                dict[strg] = self._todict(elem)
            else:
                dict[strg] = elem
        return dict

    def load_layer(self, layer_src, layer_dest, shape=None):

        layer_src = self._todict(layer_src)
        convb = np.array(layer_src['biases'], dtype=np.float32)
        convW = np.array(layer_src['filters'], dtype=np.float32)

        if shape is not None:
            convW = np.reshape(convW, shape)

        layer_dest.set_weights((convW, convb))

