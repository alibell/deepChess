from deepChess import stockFish
from deepChess import chessBoard
from deepChess.model import deepChessNN

sf = stockFish.stockFish_connector("../stockfish_14.1_linux_x64_avx2/stockfish_14.1_linux_x64_avx2")

chess = chessBoard.playChess()
