import numpy as np
#import math 
class AIPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'ai'
        self.player_string = 'Player {}:ai'.format(player_number)


    def valid(self, board):
        moves = []
        for i in range(7):
            for j in reversed(range(6)):
                if board[j][i] == 0:
                    moves.append([j, i])
                    break
        return moves

    def counter(self, board, num, player_num):
        best_n  = 0
        count_v = 0
        count_d = 0
        count_h = 0
        for row in board:
                for i in range(len(row) - num + 1):
                    win = True
                    for j in range(num):
                        if row[i + j] != player_num:
                            win = False
                            break
                    if win:
                        count_h += 1
                 
        for col in board.T:
                for i in range(len(col) - num + 1):
                    win = True
                    for j in range(num):
                        if col[i + j] != player_num:
                         win = False
                         break
                    if win:
                        count_v += 1
                 
        for op in [None, np.fliplr]:
                op_board = op(board) if op else board
                d1 = np.diagonal(op_board, offset=0)
                for i in range(len(d1) - num + 1):
                    win = True
                    for j in range(num):
                        if d1[i + j] != player_num:
                            win = False
                            break
                    if win:
                        count_d += 1

                for i in range(1, board.shape[1]-num+1):
                    for offset in [i, -i]:
                        diag = np.diagonal(op_board, offset=offset)
                        for j in range(len(diag) - num + 1):
                            win = True
                            for k in range(num):
                                if diag[j + k] != player_num:
                                    win = False
                                    break
                            if win:
                                count_d += 1
          
        best_n = count_h+count_v+count_d
        #print('THIS IS FOR EVAL!!:', best_n)
        return best_n
    #def counting_no_of_mov(seld, board):
    #    pass
    def evaluation_function(self, board):
        result = 0
        player = self.player_number
        if (player == 1): 
            opponent = 2
        else: 
            opponent = 1
        result += self.counter( board, 4, player) * 1000
        result += self.counter( board, 3, player) * 100
        result += self.counter( board, 2, player) * 10
        result -= self.counter( board, 4, opponent) * 950 #To resolve the conflict where we choose to either win or not let the opponent win
        result -= self.counter( board, 3, opponent) * 100
        result -= self.counter( board, 2, opponent) * 10

        return (result)

    def minvalue(self, board,alpha,beta,depth,player, opponent):
            valid_moves = self.valid(board)
            if(depth == 4 or not valid_moves):
                return (self.evaluation_function(board))
            i = 0
            valid_moves = self.valid(board)
            while i < len(valid_moves):
                row, col = valid_moves[i]
                board[row][col] = opponent
                result = self.maxvalue(board, alpha, beta, depth + 1, player, opponent)
                beta = min(beta, result)
                board[row][col] = 0
                if beta <= alpha:
                    return beta
                i += 1
            return beta            

    def maxvalue(self, board,alpha, beta, depth, player, opponent):
            valid_moves = self.valid(board)
            if(depth == 4):
                return (self.evaluation_function(board))
            if (not valid_moves):
                return (self.evaluation_function(board))

            i = 0
            valid_moves = self.valid(board)
            while i < len(valid_moves):
                row, col = valid_moves[i]
                board[row][col] = player
                result = self.minvalue(board, alpha, beta, depth + 1, player, opponent)
                alpha = max(alpha, result)
                board[row][col] = 0
                if alpha >= beta:
                    return alpha
                i += 1
            return alpha
    
    def get_alpha_beta_move(self, board):
        values = []
        def alphabeta( board, depth, alpha, beta, player, opponent):
            for i, j in self.valid(board):
                board[i][j] = player
                alpha = max(alpha, self.minvalue(board,alpha, beta,depth + 1 , player, opponent))
                values.append((alpha,j))
                board[i][j] = 0
            maxvalue= -100000
            max_i= None
            for i in values:    
                if i[0]>maxvalue:
                    maxvalue=i[0]
                    max_i=i[1]
            for item in values:
                if maxvalue in item:
                    max_i = item[1]
                    break
            return (max_i)
        player = self.player_number
        opponent = 2 if player == 1 else 1 
        return (alphabeta(board, 0, -100000,+100000, player, opponent)) 

   
    def get_expectimax_move(self, board):
        values = []
        currentdepth=5
        def expectimax(board, depth, player, opponent):
            a = - 100000
            for i, j in self.valid(board):
                board[i][j] = player
                a = max(a, expval(board,depth - 1 , player, opponent))
                values.append((a,j))
                board[i][j] = 0

            maxvalue = -100000
            max_index = None
            for value, index in values:
                if value > maxvalue:
                    maxvalue = value
                    max_index = index

            return (max_index)
        def maxval(board, depth, player,opponent):
            valid_moves = self.valid(board)
            if (depth == 0): 
                return (self.evaluation_function(board))
            if not valid_moves:
                return (self.evaluation_function(board))
            bestValue = -100000
            for i,j in valid_moves:
                board[i][j] = player 
                val = expval(board, depth - 1, player, opponent)
                bestValue = max(bestValue, val)
                print('BEST VAL', bestValue)
            return bestValue
        def expval(board, depth, player, opponent): 
            exp=0
            valid_moves = self.valid(board)
            lengthmoves = len(valid_moves)
            #print ('LE', lengthmoves)
            if (depth == 0): 
                #print('LENGTH', lengthmoves)
                return (self.evaluation_function(board))
            if  not valid_moves:
                return (self.evaluation_function(board))

        
            exp = 0
            counter = 0
            while counter < lengthmoves:
                i, j = valid_moves[counter]
                board[i][j] = opponent 
                val = maxval(board , depth-1, player, opponent)
                exp += val
                counter += 1
            p = exp / lengthmoves
            print("THIS IS P", p)    
            return (p)

        player = self.player_number
        opponent = 2 if player == 1 else 1 
        return (expectimax(board, currentdepth , player, opponent))


class RandomPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'random'
        self.player_string = 'Player {}:random'.format(player_number)

    def get_move(self, board):
        """
        Given the current board state select a random column from the available
        valid moves.
        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them
        RETURNS:
        The 0 based index of the column that represents the next move
        """
        valid_cols = []
        for col in range(board.shape[1]):
            if 0 in board[:,col]:
                valid_cols.append(col)

        return np.random.choice(valid_cols)


class HumanPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'human'
        self.player_string = 'Player {}:human'.format(player_number)

    def get_move(self, board):

        """
        Given the current board state returns the human input for next move
        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them
        RETURNS:
        The 0 based index of the column that represents the next move
        """

        valid_cols = []
        for i, col in enumerate(board.T):
            if 0 in col:
                valid_cols.append(i)

        move = int(input('Enter your move: '))
        while move not in valid_cols:
            print('Column full, choose from:{}'.format(valid_cols))
            move = int(input('Enter your move: '))

        return move
    