import numpy as np
from numba import njit, prange

class DenseLayer:
    def __init__(self, input_units:int, output_units:int, init_weigts:str='he', l1_reg:float=0.0, l2_reg:float=0.0):
        # Инициализация весов методом He/Xavier и смещений нулями
        if init_weigts == 'he':
            self.weights = np.random.randn(input_units, output_units) * np.sqrt(2. / input_units)                   # Для равномерного распределения весов нужно 6, а не 2
        elif(init_weigts == 'xavier'):
            self.weights = np.random.randn(input_units, output_units) * np.sqrt(2. / (input_units + output_units))  # Для равномерного распределения весов нужно 6, а не 2
        else:
            raise ValueError("Параметр init_weights принимает только два параметра: he (для ReLU) и xavier (для Sigmoid, tanh)")

        self.biases = np.zeros((1, output_units))
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.input = None
        self.forward_output = None


    def forward(self, input):
        self.input = input
        self.forward_output = np.dot(input, self.weights) + self.biases
        return self.forward_output

    def backward(self, local_grad):
        # Градиенты весов и смещений
        self.weights_grad = np.dot(self.input.T, local_grad)
        self.biases_grad = np.sum(local_grad, axis=0, keepdims=True)

        # Добавление градиентов регуляризации
        if self.l1_reg > 0:
            l1_grad = self.l1_reg * np.sign(self.weights)
            self.weights_grad += l1_grad

        if self.l2_reg > 0:
            l2_grad = self.l2_reg * self.weights
            self.weights_grad += l2_grad

        # Градиент для передачи предыдущему слою
        backward_output = np.dot(local_grad, self.weights.T)

        return backward_output
    
    
    def update(self, learning_rate):
        self.weights -= learning_rate * self.weights_grad
        self.biases -= learning_rate * self.biases_grad


    def get_regularization_loss(self):
        l1_loss = self.l1_reg * np.sum(np.abs(self.weights))
        l2_loss = self.l2_reg * 0.5 * np.sum(self.weights ** 2)
        return l1_loss + l2_loss
    
class GRU:
    def __init__(self, input_size:int, hidden_size:int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.forward_output = None

        # Инициализация параметров GRU
        self._init_parameters()

        # Кэш для обратного распространения
        self.cache = {}

    def _init_parameters(self):
        scale = np.sqrt(2.0 / (self.hidden_size + self.input_size))

        # Веса для update gate (z)
        self.Wz = np.random.uniform(-scale, scale, (self.hidden_size, self.input_size))
        self.Uz = np.random.uniform(-scale, scale, (self.hidden_size, self.hidden_size))
        self.bz = np.zeros((self.hidden_size, 1))

        # Веса для reset gate (r)
        self.Wr = np.random.uniform(-scale, scale, (self.hidden_size, self.input_size))
        self.Ur = np.random.uniform(-scale, scale, (self.hidden_size, self.hidden_size))
        self.br = np.zeros((self.hidden_size, 1))

        # Веса для candidate activation (h_tilde)
        self.Wh = np.random.uniform(-scale, scale, (self.hidden_size, self.input_size))
        self.Uh = np.random.uniform(-scale, scale, (self.hidden_size, self.hidden_size))
        self.bh = np.zeros((self.hidden_size, 1))

    def forward(self, x, h_prev = None):
        batch_size, seq_len, _ = x.shape

        # Сохраняем вход для обратного прохода
        self.cache['x'] = x
        self.cache['batch_size'] = batch_size
        self.cache['seq_len'] = seq_len

        h_states, h_current, z_gates, r_gates, h_tilde_list, h_prev_ = self._forward(batch_size, seq_len, self.hidden_size, h_prev, x, self.Wz, self.Uz, self.bz, self.Wr, self.Ur, self.br, self.Wh, self.Uh, self.bh)
        
        # Сохраняем для обратного прохода
        self.cache['z_gates'] = z_gates
        self.cache['r_gates'] = r_gates
        self.cache['h_tilde_list'] = h_tilde_list
        self.cache['h_states'] = h_states
        self.cache['h_prev'] = h_prev_

        self.forward_output = np.stack(h_states, axis=1)

        return self.forward_output, h_current

    @staticmethod
    @njit()
    def _forward(batch_size, seq_len, hidden_size, h_prev, x, Wz, Uz, bz, Wr, Ur, br, Wh, Uh, bh):
        # Инициализация скрытого состояния
        if h_prev is None:
            h_prev = np.zeros((batch_size, hidden_size))

        else:
            prev_size = h_prev.shape[-1]

            if(prev_size < hidden_size):
                padding = np.zeros((batch_size, hidden_size - prev_size))       # Паддинг нулями
                h_prev = np.concatenate((h_prev, padding), axis=-1)

            elif(prev_size > hidden_size):
                h_prev = h_prev[:, :hidden_size]        # Усечение до hidden_size

            else:
                h_prev = h_prev

        # Списки для хранения промежуточных значений
        h_states = []
        z_gates = []
        r_gates = []
        h_tilde_list = []

        h_current = h_prev

        # Проход по временной последовательности
        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch_size, input_size)

            # Вычисление гейтов
            z = 1.0 / (1.0 + np.exp(-(x_t @ Wz.T + h_current @ Uz.T + bz.T)))   #Sigmoid по z   
            r = 1.0 / (1.0 + np.exp(-(x_t @ Wr.T + h_current @ Ur.T + br.T)))   #Sigmoid по r
            h_tilde = np.tanh(x_t @ Wh.T + (r * h_current) @ Uh.T + bh.T)       #Tanh по h

            # Обновление скрытого состояния
            h_current = (1 - z) * h_current + z * h_tilde

            # Сохраняем промежуточные значения
            z_gates.append(z)
            r_gates.append(r)
            h_tilde_list.append(h_tilde)
            h_states.append(h_current)

        return h_states, h_current, z_gates, r_gates, h_tilde_list, h_prev  # (batch_size, seq_len, hidden_size) - для np.stack()

    def backward(self, upstream_grad):
        self.backward_output = None
        # Получаем данные из кэша
        x = self.cache['x']
        batch_size = self.cache['batch_size']
        seq_len = self.cache['seq_len']
        z_gates = self.cache['z_gates']
        r_gates = self.cache['r_gates']
        h_tilde_list = self.cache['h_tilde_list']
        h_states = self.cache['h_states']
        h_prev = self.cache['h_prev']

        self.backward_output, self.dWz, self.dUz, self.dbz, self.dWr, self.dUr, self.dbr, self.dWh, self.dUh, self.dbh = self._backward(upstream_grad, x, batch_size, self.hidden_size, seq_len, z_gates, r_gates, h_tilde_list, h_states, h_prev, self.Wz, self.Uz, self.bz, self.Wr, self.Ur, self.br, self.Wh, self.Uh, self.bh)
        return self.backward_output

    
    @staticmethod
    @njit()
    def _backward(upstream_grad, x, batch_size, hidden_size, seq_len, z_gates, r_gates, h_tilde_list, h_states, h_prev, Wz, Uz, bz, Wr, Ur, br, Wh, Uh, bh):
        # Инициализация градиентов параметров
        dWz, dUz, dbz = np.zeros_like(Wz), np.zeros_like(Uz), np.zeros_like(bz)
        dWr, dUr, dbr = np.zeros_like(Wr), np.zeros_like(Ur), np.zeros_like(br)
        dWh, dUh, dbh = np.zeros_like(Wh), np.zeros_like(Uh), np.zeros_like(bh)

        # Градиент по входным данным
        d_x = np.zeros_like(x)

        # Градиент по скрытому состоянию
        d_h_next = np.zeros((batch_size, hidden_size))

        # Обратный проход по времени (BPTT)
        for t in range(seq_len-1, -1, -1):
            # Получаем кэшированные значения
            x_t = x[:, t, :]
            z_t = z_gates[t]
            r_t = r_gates[t]
            h_tilde = h_tilde_list[t]

            # Предыдущее скрытое состояние
            if t == 0:
                h_prev_t = h_prev
            else:
                h_prev_t = h_states[t-1]

            # Добавляем градиент от текущего выхода
            d_h_next += upstream_grad[:, t, :]

            # Градиенты по компонентам GRU
            # Градиент по update gate
            d_z = d_h_next * (h_tilde - h_prev_t) * z_t * (1 - z_t)

            # Градиент по candidate activation
            d_h_tilde = d_h_next * z_t * (1 - h_tilde**2)

            # Градиент по reset gate
            d_r = (d_h_tilde @ Uh) * h_prev_t * r_t * (1 - r_t)

            # Градиент по предыдущему скрытому состоянию
            d_h_prev = (d_h_next * (1 - z_t) +
                       d_z @ Uz +
                       d_r @ Ur +
                       d_h_tilde @ Uh * r_t)

            # Градиенты параметров update gate
            dWz += d_z.T @ x_t
            dUz += d_z.T @ h_prev_t
            dbz += np.sum(d_z, axis=0)[:, np.newaxis]

            # Градиенты параметров reset gate
            dWr += d_r.T @ x_t
            dUr += d_r.T @ h_prev_t
            dbr += np.sum(d_r, axis=0)[:, np.newaxis]

            # Градиенты параметров candidate activation
            dWh += d_h_tilde.T @ x_t
            dUh += d_h_tilde.T @ (r_t * h_prev_t)
            dbh += np.sum(d_h_tilde, axis=0)[:, np.newaxis]

            # Градиент по входным данным для текущего временного шага
            d_x[:, t, :] = d_z @ Wz + d_r @ Wr + d_h_tilde @ Wh

            # Передаем градиент к предыдущему временному шагу
            d_h_next = d_h_prev

        return d_x, dWz, dUz, dbz, dWr, dUr, dbr, dWh, dUh, dbh
    
    def update(self, learning_rate):
        self.Wz -= learning_rate * self.dWz
        self.Uz -= learning_rate * self.dUz
        self.bz -= learning_rate * self.dbz

        self.Wr -= learning_rate * self.dWr
        self.Ur -= learning_rate * self.dUr
        self.br -= learning_rate * self.dbr

        self.Wh -= learning_rate * self.dWh
        self.Uh -= learning_rate * self.dUh
        self.bh -= learning_rate * self.dbh

class Flatten:
    def __init__(self, mode:str='last'):
        self.mode = mode
        self.cache = {}
        self.forward_output = None

    def forward(self, x):
        self.cache['input_shape'] = x.shape

        if self.mode == 'last':
            self.forward_output = x[:, -1, :]  # for sequences
        elif self.mode == 'mean':
            self.forward_output = np.mean(x, axis=1)  # for sequences
        elif self.mode == 'sum':
            self.forward_output = np.sum(x, axis=1)  # for sequences
        elif self.mode == 'cnn':
            batch_size = x.shape[0]
            self.forward_output = x.reshape(batch_size, -1)  # flatten to (batch_size, samples)
        else:
            raise ValueError("Unsupported mode")

        return self.forward_output

    def backward(self, upstream_grad):
        self.backward_output = None
        input_shape = self.cache['input_shape']

        if self.mode == 'last':
            batch_size, seq_len, features = input_shape
            self.backward_output = np.zeros((batch_size, seq_len, features))
            self.backward_output[:, -1, :] = upstream_grad
        elif self.mode == 'mean':
            batch_size, seq_len, features = input_shape
            self.backward_output = np.ones((batch_size, seq_len, features)) * upstream_grad[:, np.newaxis, :] / seq_len
        elif self.mode == 'sum':
            batch_size, seq_len, features = input_shape
            self.backward_output = np.ones((batch_size, seq_len, features)) * upstream_grad[:, np.newaxis, :]
        elif self.mode == 'cnn':
            self.backward_output = upstream_grad.reshape(input_shape)  # reshape back
        else:
            raise ValueError("Unsupported mode")

        return self.backward_output

class ReLU:
    def __init__(self):
        self.input = None  # Кэш для derivative

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def forward(self, input):
        self.input = input
        return self.relu(input)

    def backward(self, upstream_grad):
        self.backward_output = None
        activation_derivative = self.relu_derivative(self.input)
        self.backward_output = upstream_grad * activation_derivative
        return self.backward_output

class Sigmoid:
    def __init__(self):
        self.output = None

    def sigmoid(self, x):
        # Для численной стабильности
        x = np.clip(x, -500, 500)  # Предотвращаем переполнение
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, input):
        self.output = self.sigmoid(input)
        return self.output

    def backward(self, upstream_grad):
        self.backward_output = None
        activation_derivative = self.sigmoid_derivative(self.output)
        self.backward_output = upstream_grad * activation_derivative
        return self.backward_output
    
class BatchNormalization:
    def __init__(self, num_features:int, eps=1e-5, momentum=0.9):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Обучаемые параметры
        self.gamma = np.ones((1, num_features))  # scale
        self.beta = np.zeros((1, num_features))  # shift

        self.dgamma = None
        self.dbeta = None

        # Running статистика (для инференса)
        self.running_mean = np.zeros((1, num_features))
        self.running_var = np.ones((1, num_features))

        # Кэш для backward pass
        self.cache = None

    def forward(self, x, training:bool=True):
        if training:
            #Вычисление среднего по батчу (mean по оси 0)
            mu = np.mean(x, axis=0, keepdims=True)  # shape (1, num_features)

            #Вычисление дисперсии по батчу
            var = np.var(x, axis=0, keepdims=True)  # shape (1, num_features)

            # Нормализация
            x_hat = (x - mu) / np.sqrt(var + self.eps)

            #Scale и shift
            y = self.gamma * x_hat + self.beta

            #Обновление running mean и var
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var

            # Сохранение в кэш для backward
            self.cache = (x, mu, var, x_hat)

            return y
        else:
            # Инференс: использование running статистики
            x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            y = self.gamma * x_hat + self.beta
            self.cache = (x, x_hat)
            return y

    def backward(self, dy):
        self.backward_output = None
        if (len(self.cache) == 2):
            x, x_hat = self.cache
            mu, var = self.running_mean, self.running_var
            #raise ValueError("Сначала должен быть запущен прямой проход - потом обратный!")
        else:
            x, mu, var, x_hat = self.cache
        batch_size = x.shape[0]

        # Градиент по gamma
        self.dgamma = np.sum(dy * x_hat, axis=0, keepdims=True)

        # Градиент по beta
        self.dbeta = np.sum(dy, axis=0, keepdims=True)

        # Градиент по x_hat
        dx_hat = dy * self.gamma

        # Градиент по var
        dvar = np.sum(dx_hat * (x - mu) * (-0.5) * (var + self.eps) ** (-1.5), axis=0, keepdims=True)

        # Градиент по mu
        dmu = np.sum(dx_hat * (-1 / np.sqrt(var + self.eps)), axis=0, keepdims=True)

        # Градиент по x
        self.backward_output = (dx_hat / np.sqrt(var + self.eps)) + \
             (dvar * (2 * (x - mu) / batch_size)) + \
             (dmu / batch_size)
        
        return self.backward_output
        
    def update(self, learning_rate):
        self.gamma -= learning_rate * self.dgamma
        self.beta -= learning_rate * self.dbeta

class Dropout:
    def __init__(self, p=0.5):
        self.p = p

    def forward(self, X, train:bool=True):
        if not train:
            self.mask = np.ones((X.shape))
            return X
        self.mask = (np.random.rand(*X.shape) > self.p) / (1.0 - self.p)
        return X * self.mask

    def backward(self, upstream_grad):
        return upstream_grad * self.mask
    
class MinPooling2D:
    def __init__(self, pool_size=(2, 2), strides=(2, 2), padding:str='valid'):
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding
        self.input = None
        self.min_indices = None
        self.forward_output = None

    def forward(self, input):
        self.input = input
        batch_size, channels, height, width = input.shape

        pool_height, pool_width = self.pool_size
        stride_height, stride_width = self.strides

        if self.padding == 'valid':
            pad_height = 0
            pad_width = 0
        elif self.padding == 'same':
            pad_height = (stride_height * (height - 1) + pool_height - height) // 2
            pad_width = (stride_width * (width - 1) + pool_width - width) // 2
        else:
            raise ValueError("Padding может быть 'valid' или 'same'")

        padded_input = np.pad(input, ((0, 0), (0, 0), (pad_height, pad_height), (pad_width, pad_width)), mode='constant')

        out_height = (height + 2 * pad_height - pool_height) // stride_height + 1
        out_width = (width + 2 * pad_width - pool_width) // stride_width + 1

        self.forward_output, self.min_indices = self.__forward(out_height, out_width, batch_size, channels, stride_height, stride_width, pool_height, pool_width, padded_input)

        return np.array(self.forward_output)

    @staticmethod
    @njit()
    def __forward(out_height, out_width, batch_size, channels, stride_height, stride_width, pool_height, pool_width, padded_input):
      output = [[[[0.0 for _ in range(out_width)] for _ in range(out_height)] for _ in range(channels)] for _ in range(batch_size)]
      min_indices = [[[[[0, 0] for _ in range(out_width)] for _ in range(out_height)] for _ in range(channels)] for _ in range(batch_size)]

      for b in prange(batch_size):
          for c in prange(channels):
              for i in range(out_height):
                  for j in range(out_width):
                      h_start = i * stride_height
                      w_start = j * stride_width

                      # Извлечение "патча"
                      patch = [row[w_start:w_start + pool_width] for row in padded_input[b][c][h_start:h_start + pool_height]]

                      # Поиск минимального значения и его индекса
                      min_val = float('inf')
                      min_i, min_j = 0, 0
                      for pi in range(len(patch)):
                          for pj in range(len(patch[pi])):
                              val = patch[pi][pj]
                              if val < min_val:
                                  min_val = val
                                  min_i, min_j = pi, pj

                      output[b][c][i][j] = min_val
                      min_indices[b][c][i][j] = [h_start + min_i, w_start + min_j]

      return output, min_indices


    def backward(self, upstream_grad):
        batch_size, channels, out_height, out_width = upstream_grad.shape

        return self.__backward(np.array(self.input), self.strides, self.padding, self.pool_size, np.array(self.min_indices), out_height, out_width, batch_size, channels, np.array(upstream_grad))


    @staticmethod
    @njit()
    def __backward(input, strides, padding, pool_size, min_indices, out_height, out_width, batch_size, channels, upstream_grad):
        backward_output = np.zeros_like(input)

        pad_height = (input.shape[2] - (out_height - 1) * strides[0] - 1 + pool_size[0]) // 2 if padding == 'same' else 0
        pad_width = (input.shape[3] - (out_width - 1) * strides[1] - 1 + pool_size[1]) // 2 if padding == 'same' else 0

        for b in prange(batch_size):
            for c in prange(channels):
                for i in range(out_height):
                    for j in range(out_width):
                        h, w = min_indices[b][c][i][j]
                        backward_output[b][c][h - pad_height][w - pad_width] += upstream_grad[b][c][i][j]

        return backward_output
    
class Conv_2D:
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int, stride:int=1, padding:int=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.weights = (np.random.randn(out_channels, in_channels, *self.kernel_size).astype(np.float32) * 0.01)
        self.bias = np.zeros((out_channels,), dtype=np.float32)

    def forward(self, x):
        self.x = x
        batch_size, _, in_h, in_w = x.shape
        k_h, k_w = self.kernel_size
        s = self.stride
        p = self.padding
        out_h = (in_h + 2 * p - k_h) // s + 1
        out_w = (in_w + 2 * p - k_w) // s + 1
        out_shape = (batch_size, self.out_channels, out_h, out_w)
        return self._conv_forward(x, self.weights, self.bias, s, p, out_shape)

    def backward(self, d_out):
        self.backward_output = None
        x = self.x
        self.batch_size, _, _, _ = x.shape
        s = self.stride
        p = self.padding
        d_w_shape = self.weights.shape
        d_b_shape = self.bias.shape
        d_x_shape = x.shape
        self.d_w, self.d_b, self.backward_output = self._conv_backward(x, self.weights, d_out, s, p, d_w_shape, d_b_shape, d_x_shape)
        return self.backward_output
    
    def update(self, learning_rate):
        self.weights -= learning_rate * (self.d_w / self.batch_size)
        self.bias -= learning_rate * (self.d_b / self.batch_size)

    @staticmethod
    @njit()
    def _conv_forward(x, weights, bias, stride, padding, out_shape):
        batch_size, _, in_h, in_w = x.shape
        out_channels, _, k_h, k_w = weights.shape
        out_h, out_w = out_shape[2], out_shape[3]
        out = np.zeros(out_shape, dtype=x.dtype)
        for b in prange(batch_size):
            for oc in range(out_channels):
                for i in range(out_h):
                    h_start = i * stride - padding
                    h_end = h_start + k_h
                    for j in range(out_w):
                        w_start = j * stride - padding
                        w_end = w_start + k_w
                        h_s = max(h_start, 0)
                        h_e = min(h_end, in_h)
                        w_s = max(w_start, 0)
                        w_e = min(w_end, in_w)
                        if h_s >= h_e or w_s >= w_e:
                            out[b, oc, i, j] = bias[oc]
                            continue
                        wh_s = h_s - h_start
                        ww_s = w_s - w_start
                        h_len = h_e - h_s
                        w_len = w_e - w_s
                        input_slice = x[b, :, h_s:h_e, w_s:w_e]
                        weight_slice = weights[oc, :, wh_s:wh_s + h_len, ww_s:ww_s + w_len]
                        out[b, oc, i, j] = np.sum(input_slice * weight_slice) + bias[oc]
        return out

    @staticmethod
    @njit()
    def _conv_backward(x, weights, d_out, stride, padding, d_w_shape, d_b_shape, d_x_shape):
        batch_size, _, in_h, in_w = x.shape
        out_channels, _, k_h, k_w = weights.shape
        _, _, out_h, out_w = d_out.shape
        d_w = np.zeros(d_w_shape, dtype=weights.dtype)
        d_b = np.zeros(d_b_shape, dtype=weights.dtype)
        d_x = np.zeros(d_x_shape, dtype=x.dtype)
        for b in prange(batch_size):
            for oc in range(out_channels):
                for i in range(out_h):
                    h_start = i * stride - padding
                    h_end = h_start + k_h
                    for j in range(out_w):
                        w_start = j * stride - padding
                        w_end = w_start + k_w
                        h_s = max(h_start, 0)
                        h_e = min(h_end, in_h)
                        w_s = max(w_start, 0)
                        w_e = min(w_end, in_w)
                        grad_scalar = d_out[b, oc, i, j]
                        if h_s >= h_e or w_s >= w_e:
                            d_b[oc] += grad_scalar
                            continue
                        wh_s = h_s - h_start
                        ww_s = w_s - w_start
                        h_len = h_e - h_s
                        w_len = w_e - w_s
                        input_slice = x[b, :, h_s:h_e, w_s:w_e]
                        weight_slice = weights[oc, :, wh_s:wh_s + h_len, ww_s:ww_s + w_len]
                        d_w[oc, :, wh_s:wh_s + h_len, ww_s:ww_s + w_len] += grad_scalar * input_slice
                        d_x[b, :, h_s:h_e, w_s:w_e] += grad_scalar * weight_slice
                        d_b[oc] += grad_scalar
        return d_w, d_b, d_x
    
class Concatenate:
    def __init__(self, axis=-1):
        self.axis = axis
        self.cache = {}
        self.forward_output = None

    def forward(self, inputs):
        # Проверка что все массивы имеют одинаковую размерность
        num_dims = len(inputs[0].shape)
        for i, x in enumerate(inputs):
            if len(x.shape) != num_dims:
                raise ValueError(f"Все входы должны иметь одинаковую размерность. "
                               f"Вход 0: {num_dims} измерений, вход {i}: {len(x.shape)} измерений")

        # Нормализация оси (для отрицательных значений)
        axis = self.axis if self.axis >= 0 else num_dims + self.axis

        # Проверка совместимости форм (кроме выбранной оси)
        for i in range(1, len(inputs)):
            for dim in range(num_dims):
                if dim != axis and inputs[0].shape[dim] != inputs[i].shape[dim]:
                    raise ValueError(f"Все входы должны иметь одинаковую форму кроме оси конкатенации. "
                                   f"Вход 0: {inputs[0].shape}, вход {i}: {inputs[i].shape}")

        # Выполнение конкатенации
        self.forward_output = np.concatenate(inputs, axis=axis)

        # Сохранение данных для обратного прохода
        self.cache['input_shapes'] = [x.shape for x in inputs]
        self.cache['axis'] = axis
        self.cache['num_inputs'] = len(inputs)

        return self.forward_output

    def backward(self, upstream_grad):
        self.backward_output = None
        input_shapes = self.cache['input_shapes']
        axis = self.cache['axis']

        # Вычисление размеров вдоль оси конкатенации для каждого входа
        sizes = [shape[axis] for shape in input_shapes]

        # Вычисление индексов для разделения
        if len(sizes) > 1:
            indices = np.cumsum(sizes)[:-1]  # Исключаем последний индекс
        else:
            indices = []  # Для одного входа

        # Разделение градиента по исходным входам
        self.backward_output = np.split(upstream_grad, indices, axis=axis)

        return self.backward_output
    