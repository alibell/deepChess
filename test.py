from deepChess import stockFish

opponent = stockFish.stockFish_connector("../stockfish_14.1_linux_x64_avx2/stockfish_14.1_linux_x64_avx2")

print(
    opponent.get_board()
)


