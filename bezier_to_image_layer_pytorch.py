import torch


class Bezier2Image(torch.nn.Module):
    def __init__(self, n=30, w=60, length=160, alpha=0.0002, viewbox=(0, 0, 1.0, 1.0)):

        super().__init__()
        self.n = n
        self.w = w
        self.length = length
        self.viewbox = viewbox
        self.alpha = alpha

        t = torch.arange(0, self.n, dtype=torch.float32) / self.n
        # y=2x^3-3x^2+2xに沿って密度を変更することで、より美しく描画ができる。この操作は計算量に影響しない。
        t = 2*torch.pow(t, 3)-3*torch.square(t)+2*t
        t_bar = 1 - t
        t_3_0 = torch.pow(t, 3)
        t_2_1 = torch.square(t) - t_3_0
        t_1_2 = t_3_0 - 2*torch.square(t) + t
        t_0_3 = torch.pow(t_bar, 3)

        self.T = torch.stack(
            [t_3_0, 3*t_2_1, 3*t_1_2, t_0_3], axis=1)  # shape=(n, 4)

        board = torch.arange(
            0, self.w, dtype=torch.float32) / self.w  # shape=(w)
        board = torch.unsqueeze(board, -1)  # shape=(w, 1)
        board = board.repeat((1, self.n))  # shape=(w, n)
        board = torch.unsqueeze(board, axis=0)  # shape=(1, w, n)
        board = board.repeat((self.length, 1, 1))  # shape=(length, w, n)
        board = torch.unsqueeze(board, axis=0)  # shape=(1, length, w, n)
        self.board_X = board*self.viewbox[2] + self.viewbox[0]
        self.board_Y = board*self.viewbox[3] + self.viewbox[1]

    def forward(self, x):
        """
        入力されるベジェ曲線は(None, length, 8)のデータ列になっている。
        """
        # 1.通る点を求める(nは曲線を何分割するか)
        reshaped_bezier = torch.reshape(
            x, (-1, x.shape[1], 4, 2))  # shape=(None, length, 4, 2)
        points = torch.matmul(self.T, reshaped_bezier)
        # shape=(None, length, n, 2)
        points = torch.reshape(points, (-1, self.length, self.n, 2))

        # 2.求めたものを画像に変換していく。
        X = points[:, :, :, 0].unsqueeze(-1)
        X = torch.permute(X, (0, 1, 3, 2))
        Y = points[:, :, :, 1].unsqueeze(-1)
        Y = torch.permute(Y, (0, 1, 3, 2))

        board_X = torch.exp(-torch.square((self.board_X-X))/self.alpha)
        board_Y = torch.exp(-torch.square((self.board_Y-Y))/self.alpha)
        board_Y_T = torch.permute(board_Y, (0, 1, 3, 2))

        result = torch.matmul(board_X, board_Y_T)  # shape = (None, 160, w, w)
        result = torch.sum(result, axis=1)
        result = torch.minimum(
            result, torch.ones_like(result))  # (画像フォーマットとしては必要)

        return result


# Create Tensors to hold input and outputs.
input = torch.Tensor([
    [0.0, 0.0, 0.3, 0.3, 0.6, 0.3, 0.9, 0.0],
    [0.9, 0.0, 1.2, 0.3, 1.2, 0.6, 0.9, 0.9],
    [0.9, 0.9, 0.6, 1.2, 0.3, 1.2, 0.0, 0.9],
    [0.0, 0.9, -0.3, 0.6, -0.3, 0.3, 0.0, 0.0],
]).unsqueeze(0)

# Construct our model by instantiating the class defined above
model = Bezier2Image(length=4, w=50, n=60, alpha=0.0004,
                     viewbox=(-1, -1, 2, 2))
y_pred = model(input)
