import keras
from keras import backend as K 
from keras.engine.topology import Layer

class BezierToImageLayer(Layer):
  def __init__(self, n=30, w=60, length=160, alpha=0.0002, viewbox=(0,0,1.0,1.0), **kwargs):
    super(BezierToImageLayer, self).__init__(**kwargs)
    self.n = n
    self.w = w
    self.length = length
    self.viewbox = viewbox
    self.alpha = alpha
    t = K.arange(0, self.n, dtype="float32")/self.n
    t = 2*K.pow(t,3)-3*K.square(t)+2*t  #y=2x^3-3x^2+2xに沿って密度を変更することで、より美しく描画ができる。この操作は計算量に影響しない。
    t_bar = 1 - t
    t_3_0 = K.pow(t, 3)
    t_2_1 = K.square(t) - t_3_0
    t_1_2 = t_3_0 - 2*K.square(t) + t
    t_0_3 = K.pow(t_bar, 3)

    self.T = K.stack([t_3_0, 3*t_2_1, 3*t_1_2, t_0_3], axis=1) #shape=(n, 4)

    board = K.arange(0,self.w, dtype='float32')/self.w                #shape=(w)
    board = K.expand_dims(board)                                      #shape=(w, 1)
    board = K.repeat_elements(board, self.n, axis=1)                  #shape=(w, n)
    board = K.expand_dims(board, axis=0)                              #shape=(1, w, n)
    board = K.repeat_elements(board, self.length, axis=0)             #shape=(length, w, n)
    board = K.expand_dims(board, axis=0)                              #shape=(1, length, w, n)
    self.board_X = board*self.viewbox[2] + self.viewbox[0]
    self.board_Y = board*self.viewbox[3] + self.viewbox[1]

  def build(self, input_shape):  
    super(BezierToImageLayer, self).build(input_shape)  # Be sure to call this somewhere!

  def call(self, x):
    """
    入力されるベジェ曲線は(None, length, 8)のデータ列になっている。
    """
    #1.通る点を求める(nは曲線を何分割するか)
    reshaped_bezier = K.reshape(x, (-1, x.shape[1], 4, 2))   #shape=(None, length, 4, 2)
    points = tf.matmul(self.T, reshaped_bezier)
    points = K.reshape(points, (-1, self.length, self.n, 2))          #shape=(None, length, n, 2)

    #2.求めたものを画像に変換していく。
    X = K.expand_dims(points[:, :, :, 0])                             #shape = None, length, n
    X = K.permute_dimensions(X, (0,1,3,2))
    Y = K.expand_dims(points[:, :, :, 1])
    Y = K.permute_dimensions(Y, (0,1,3,2))

    board_X = K.exp(-K.square((self.board_X-X))/self.alpha)
    board_Y = K.exp(-K.square((self.board_Y-Y))/self.alpha)
    board_Y_T = K.permute_dimensions(board_Y, (0,1,3,2))

    result = tf.matmul(board_X, board_Y_T) #shape = (None, 160, w, w)
    result = K.sum(result, axis=1)
    result = K.minimum(result, K.ones_like(result))    #(画像フォーマットとしては必要)

    return result

  def compute_output_shape(self, input_shape):
      return (input_shape[0], self.w, self.w)
