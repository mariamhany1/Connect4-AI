# Nouran Ahmed Abdelaziz 20200609
# Mariam Hany Gamal      20200532
# Fatma Salah Mahmoud    20200376
# Nada Ashraf Mahmoud    20200587
# Ziyad Ashraf Azab      20200197

from tkinter import *
import numpy as np
import random
import pygame
import sys
import math
import copy
from timeit import default_timer as timer

# The colors of pygame gui (the board)
BLUE = (50, 0, 255)
WHITE = (255, 240, 255)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
GREEN = (255, 100, 0)

# global variables for the row and col number of the board
ROWNUM = 6
COLNUM = 7

COMPUTER = 0
AI_AGENT = 1

EMPTY = 0
COMPUTER_MARK = 1
AI_MARK = 2

maximizingPlayer = 1
minimizingPlayer = 0

WINDOW_SIZE = 4


# pygame board
EACH_PIECE_SIZE = 100  # the size of each piece in the board
width = COLNUM * EACH_PIECE_SIZE
height = (ROWNUM + 1) * EACH_PIECE_SIZE
size = (width, height)
RADIUS = int(EACH_PIECE_SIZE / 2 - 5)
screen = pygame.display.set_mode(size)


# function to be called when press the difficult minimax button in gui
def runDifficultMinimax():
    board = generateBoard()
    printBoard(board)
    game_over = False  # to stop the game after winning

    start = timer() 
    pygame.init()

    draw_board(board)
    pygame.display.update()

    myfont = pygame.font.SysFont("monospace", 75)

    # to start the game randomly by any player
    turn = random.randint(COMPUTER, AI_AGENT)

    # while still no one wins
    while not game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            pygame.display.update()
        # if it's the computer turn to play (using minimax algorithm)
        if turn == COMPUTER:
            col, scorer = minimax(board, 3, True)
            # if the board is full and no one wins
            if BoardFull(board):
                label = myfont.render("DRAW GAME", 1, GREEN)
                screen.blit(label, (40, 10))
                game_over = True  # to stop the game
            # check if it's a valid column to place a piece inside its deepest row
            # by using the column that minimax algorithm returned (the best col to place a piece in it)
            elif isValidCol(board, col):
                pygame.time.wait(700)
                row = findDeepestRow(board, col)
                placeTiles(board, row, col, COMPUTER_MARK)

                # after placing the piece we check if the computer win or not
                # and our priority to make the computer loses and the agent wins
                if isWinning(board, COMPUTER_MARK):
                    label = myfont.render("AGENT 1 WINS", 1, RED)
                    screen.blit(label, (40, 10))
                    game_over = True # to stop the game

                # each time we place a piece we print the board to check if it places it in the board correctly
                printBoard(board)
                draw_board(board)

                # to change turns between computer and AI agent
                turn = AI_AGENT

        # if it's the AI_AGENT turn to play (using minimax algorithm)
        elif turn == AI_AGENT and not game_over:
            # depth is 5, more than the depth in the Computer turn to prioritize its winning
            col, scorer = minimax(board, 4, True)
            # if the board is full and no one wins
            if BoardFull(board):
                label = myfont.render("DRAW GAME", 1, GREEN)
                screen.blit(label, (40, 10))
                game_over = True # to stop the game
                printBoard(board)
                draw_board(board)
            # check if it's a valid column to place a piece inside its deepest row
            # by using the column that minimax algorithm returned (the best col to place a piece in it)
            elif isValidCol(board, col):
                pygame.time.wait(700)
                row = findDeepestRow(board, col)
                placeTiles(board, row, col, AI_MARK)

                # after placing the piece we check if the ai_agent win or not
                if isWinning(board, AI_MARK):
                    label = myfont.render("AGENT 2 WINS", 1, YELLOW)
                    screen.blit(label, (40, 10))
                    game_over = True # to stop the game

                printBoard(board)
                draw_board(board)

                # to change turns between computer and AI agent
                turn = COMPUTER

        if game_over:
            pygame.time.wait(3000)
    end = timer()
    time_taken = end - start                
    print("Difficult Minimax performance: ", time_taken)



# function to be called when press the easy minimax button in gui
def runEasyMinimax():
    board = generateBoard()
    printBoard(board)
    game_over = False

    start = timer() 
    pygame.init()
    draw_board(board)
    pygame.display.update()

    myfont = pygame.font.SysFont("monospace", 75)

    # to start the game randomly by any player
    turn = random.randint(COMPUTER, AI_AGENT)

    # while still no one wins
    while not game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            pygame.display.update()

        # if it's the computer turn to play
        if turn == COMPUTER:
            # we choose a random column to play a red piece in
            col = random.randint(0, 6)
            if BoardFull(board):
                label = myfont.render("DRAW GAME", 1, GREEN)
                screen.blit(label, (40, 10))
                game_over = True
            # if the random column is valid, we find the deepest availabe row to place a piece in
            elif isValidCol(board, col):
                pygame.time.wait(700)
                row = findDeepestRow(board, col)
                placeTiles(board, row, col, COMPUTER_MARK)

                #if the computer wins (by randomly choosing any col to play in)
                if isWinning(board, COMPUTER_MARK):
                    label = myfont.render("AGENT 1 WINS", 1, RED)
                    screen.blit(label, (40, 10))
                    game_over = True
                turn = AI_AGENT

                printBoard(board)
                draw_board(board)

        # if it's the ai_agent's turn to play
        elif turn == AI_AGENT and not game_over:
            # the ai_agent plays using minimax algorithm because AI agent is more intelligent
            # than the computer
            col, scorer = minimax(board, 4, True)
            #if the board is full with pieces and no one has won yet
            if BoardFull(board):
                label = myfont.render("DRAW GAME", 1, GREEN)
                screen.blit(label, (40, 10))
                game_over = True
            # if the col the minimax returned is valid, we place a piece in its deepest row
            elif isValidCol(board, col):
                pygame.time.wait(700)
                row = findDeepestRow(board, col)
                placeTiles(board, row, col, AI_MARK)

                # check if the ai_agent wins after placing the piece
                if isWinning(board, AI_MARK):
                    label = myfont.render("AGENT 2 WINS", 1, YELLOW)
                    screen.blit(label, (40, 10))
                    game_over = True

                printBoard(board)
                draw_board(board)

                turn = COMPUTER
        
        if game_over:
            pygame.time.wait(3000)
    end = timer()
    time_taken = end - start                
    print("Easy Minimax performance: ", time_taken)

# function to be called when press the easy alpha-beta minimax button in gui
def runEasyAlphaBetaMinimax():
    board = generateBoard()
    printBoard(board)
    game_over = False

    start = timer() 
    pygame.init()
    draw_board(board)
    pygame.display.update()

    myfont = pygame.font.SysFont("monospace", 75)

    # to start the game randomly by any player
    turn = random.randint(COMPUTER, AI_AGENT)

    # while still no one wins
    while not game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            pygame.display.update()

        # if it's the computer turn to play
        if turn == COMPUTER:
            # we choose a random column to play a red piece in
            col = random.randint(0, 6)
            if BoardFull(board):
                label = myfont.render("DRAW GAME", 1, GREEN)
                screen.blit(label, (40, 10))
                game_over = True
            # if the random column is valid, we find the deepest availabe row to place a piece in
            elif isValidCol(board, col):
                pygame.time.wait(700)
                row = findDeepestRow(board, col)
                placeTiles(board, row, col, COMPUTER_MARK)

                #if the computer wins (by randomly choosing any col to play in)
                if isWinning(board, COMPUTER_MARK):
                    label = myfont.render("AGENT 1 WINS", 1, RED)
                    screen.blit(label, (40, 10))
                    game_over = True
                turn = AI_AGENT

                printBoard(board)
                draw_board(board)

        # if it's the ai_agent's turn to play
        elif turn == AI_AGENT and not game_over:
            # the ai_agent plays using alpha-beta minimax algorithm because AI agent is more intelligent
            # than the computer
            col, scorer = minimaxAlphaBeta(board, 4,-math.inf, math.inf, True)
            #if the board is full with pieces and no one has won yet
            if BoardFull(board):
                label = myfont.render("DRAW GAME", 1, GREEN)
                screen.blit(label, (40, 10))
                game_over = True
            # if the col the minimax returned is valid, we place a piece in its deepest row
            elif isValidCol(board, col):
                pygame.time.wait(700)
                row = findDeepestRow(board, col)
                placeTiles(board, row, col, AI_MARK)

                # check if the ai_agent wins after placing the piece
                if isWinning(board, AI_MARK):
                    label = myfont.render("AGENT 2 WINS", 1, YELLOW)
                    screen.blit(label, (40, 10))
                    game_over = True

                printBoard(board)
                draw_board(board)

                turn = COMPUTER

        if game_over:
            pygame.time.wait(3000)
    end = timer()
    time_taken = end - start                
    print("Easy Minimax using alpha beta performance: ", time_taken)

# function to be called when press the difficult alpha-beta minimax button in gui
def runDifficultAlphaBetaMinimax():
    board = generateBoard()
    printBoard(board)
    game_over = False  # to stop the game after winning

    start = timer() 
    pygame.init()

    draw_board(board)
    pygame.display.update()

    myfont = pygame.font.SysFont("monospace", 75)

    # to start the game randomly by any player
    turn = random.randint(COMPUTER, AI_AGENT)

    # while still no one wins
    while not game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            pygame.display.update()
        # if it's the computer turn to play (using minimax algorithm)
        if turn == COMPUTER:
            col, scorer = minimax(board, 2, True)
            # if the board is full and no one wins
            if BoardFull(board):
                label = myfont.render("DRAW GAME", 1, GREEN)
                screen.blit(label, (40, 10))
                game_over = True  # to stop the game
            # check if it's a valid column to place a piece inside its deepest row
            # by using the column that minimax algorithm returned (the best col to place a piece in it)
            elif isValidCol(board, col):
                pygame.time.wait(700)
                row = findDeepestRow(board, col)
                placeTiles(board, row, col, COMPUTER_MARK)

                # after placing the piece we check if the computer win or not
                # and our priority to make the computer loses and the agent wins
                if isWinning(board, COMPUTER_MARK):
                    label = myfont.render("AGENT 1 WINS", 1, RED)
                    screen.blit(label, (40, 10))
                    game_over = True # to stop the game

                # each time we place a piece we print the board to check if it places it in the board correctly
                printBoard(board)
                draw_board(board)

                # to change turns between computer and AI agent
                turn = AI_AGENT

        # if it's the AI_AGENT turn to play (using alpha-beta minimax algorithm)
        elif turn == AI_AGENT and not game_over:
            # depth is 5, more than the depth in the Computer turn to prioritize its winning
            col, scorer = minimaxAlphaBeta(board, 5, -math.inf, math.inf, True)
            # if the board is full and no one wins
            if BoardFull(board):
                label = myfont.render("DRAW GAME", 1, GREEN)
                screen.blit(label, (40, 10))
                game_over = True # to stop the game
                printBoard(board)
                draw_board(board)
            # check if it's a valid column to place a piece inside its deepest row
            # by using the column that minimax algorithm returned (the best col to place a piece in it)
            elif isValidCol(board, col):
                pygame.time.wait(700)
                row = findDeepestRow(board, col)
                placeTiles(board, row, col, AI_MARK)

                # after placing the piece we check if the ai_agent win or not
                if isWinning(board, AI_MARK):
                    label = myfont.render("AGENT 2 WINS", 1, YELLOW)
                    screen.blit(label, (40, 10))
                    game_over = True # to stop the game

                printBoard(board)
                draw_board(board)

                # to change turns between computer and AI agent
                turn = COMPUTER

        if game_over:
            pygame.time.wait(3000)
    end = timer()
    time_taken = end - start                
    print("Difficult Minimax using alpha beta performance: ", time_taken)


# Function to determine the buttons selection
def determine_selection(p,t):
    if(p == "minimax" and t == "easy"):
        runEasyMinimax()
    elif(p == "minimax" and t == "difficult"):
        runDifficultMinimax()
    elif(p == "alpha-beta" and t == "easy"):
        runEasyAlphaBetaMinimax()
    elif(p == "alpha-beta" and t == "difficult"):
        runDifficultAlphaBetaMinimax()

# GUI
# Window 1
root = Tk()
root.title("CONNECT-4")
root.minsize(400, 300)

def first_window():
    
    # Window 2
    def second_window(p):
        label1.destroy()
        button_border1.destroy()
        minimax_button.destroy()
        button_border2.destroy()
        alphaBeta_button.destroy()

        # Label 2
        label2 = Label(root, text="   Choose level of difficulty   ", font=("Calibri bold", "15"),pady= 3)
        label2.pack()

        
        # Easy Button
        button_border3 = LabelFrame(root, highlightbackground = "black", highlightthickness = 3, bd=0)
        button_border3.pack(pady= 10)
        easy_button = Button(button_border3, text="Easy", font=("Calibri bold", "12"), width=25, fg="black", bg="light blue", command=lambda t = "easy":determine_selection(p,t))
        easy_button.pack()

        # Difficult Button
        button_border4 = LabelFrame(root, highlightbackground = "black", highlightthickness = 3, bd=0)
        button_border4.pack(pady= 10)
        difficult_button = Button(button_border4, text="Difficult", font=("Calibri bold", "12"), width=25, fg="black", bg="light blue", command=lambda t = "difficult": determine_selection(p,t))
        difficult_button.pack()

        #Back Button
        def back():
            label2.destroy()
            button_border3.destroy()
            easy_button.destroy()
            button_border4.destroy()
            difficult_button.destroy()
            button_border5.destroy()
            back_button.destroy()
            first_window()
        button_border5 = LabelFrame(root, highlightbackground = "black", highlightthickness = 3, bd=0)
        button_border5.pack(pady= 10)
        back_button = Button(button_border5, text="Back", font=("Calibri bold", "12"), width=25, fg="black", bg="light blue", command=back)
        back_button.pack()

    # Label 1
    label1 = Label(root, text="   Choose the algorithm   ", font=("Calibri bold", "15"),pady= 3)
    label1.pack()

    # Minimax Button
    button_border1 = LabelFrame(root, highlightbackground = "black", highlightthickness = 3, bd=0)
    button_border1.pack(pady=20)
    minimax_button = Button(button_border1, text="Minimax", font=("Calibri bold", "12"), width=25, fg="black", bg="light blue", command= lambda p = "minimax":second_window(p))
    minimax_button.pack()

    # Alpha Beta Button
    button_border2 = LabelFrame(root, highlightbackground = "black", highlightthickness = 3, bd=0)
    button_border2.pack()
    alphaBeta_button = Button(button_border2, text="Alpha-Beta", font=("Calibri bold", "12"), width=25, fg="black", bg="light blue", command=lambda p = "alpha-beta": second_window(p))
    alphaBeta_button.pack()
first_window()



# boolean function to loop over the board and check if it's full or not
def BoardFull(board):
    for i in range(ROWNUM):
        for j in range(COLNUM):
            if board[i][j] == EMPTY:
                return False
    return True


# generating a board and initialize it by zeroes
def generateBoard():
    board = np.zeros((ROWNUM, COLNUM), dtype=int)
    return board


# function to put a piece in the board instead of zero
def placeTiles(board, row, col, mark):
    board[row][col] = mark

# function to check if we arrived to the end of the game
# by checking if the COMPUTER wins or the AI Agent wins or there are no valid cols to put pieces in
def is_terminal_node(board):
    if isWinning(board, COMPUTER_MARK) or isWinning(board, AI_MARK) or len(eachValidColumn(board)) == 0:
        return True
    else:
        return False


# a boolean function to check if a col is empty or not
# then use the findDeepestRow function to place a piece in it
def isValidCol(board, col):
    if board[ROWNUM - 1][col] == 0:
        return True
    else:
        return False

# function to loop aver the board from the top to the botttom
# to place a piece in the deepest index availabe
def findDeepestRow(board, col):
    for i in range(0, ROWNUM - 1, +1):
        if board[i][col] == EMPTY:
            return i
    return -1


# function to store all the valid locations that have no pieces in it in an array
def eachValidColumn(board):
    valid_locations = []
    for col in range(COLNUM):
        if isValidCol(board, col):
            valid_locations.append(col)
    return valid_locations


# function to get the opponent of the player in turn
def getOpponent(player):
    if player == COMPUTER:
        return AI_AGENT
    else:
        return COMPUTER


# function to print the board in a reversed order
def printBoard(board):
    for row in reversed(board):
        row_str = ' '.join(map(str, row))
        print(row_str)
    print("\n")


# function to check if the player has won or not
def isWinning(board, mark):
    # Check if the player is winning horizontally
    for i in range(ROWNUM):
        for j in range(COLNUM - 3): # -3 because we check the four adjacent cells before we reach the last 3 cols
            if board[i][j] == mark and board[i][j + 1] == mark and board[i][j + 2] == mark and board[i][j + 3] == mark:
                return True
    # Check if the player is winning vertically
    for i in range(ROWNUM - 3):
        for j in range(COLNUM):
            if board[i][j] == mark and board[i + 1][j] == mark and board[i + 2][j] == mark and board[i + 3][j] == mark:
                return True
    # Check if the player is winning in a negative diagonal way
    for i in range(ROWNUM - 3):
        for j in range(COLNUM - 3):
            if board[i][j] == mark and board[i + 1][j + 1] == mark and board[i + 2][j + 2] == mark and board[i + 3][j + 3] == mark:
                return True
    # Check if the player is winning in a positive diagonal way
    for i in range(ROWNUM - 3):
        for j in range(3, COLNUM):
            if board[i][j] == mark and board[i + 1][j - 1] == mark and board[i + 2][j - 2] == mark and board[i + 3][j - 3] == mark:
                return True
    return False

# function to calculates the score for a given window of game positions based on the provided mark (player)
def calculateScore(window, mark):
    score = 0
    ai_piece = COMPUTER_MARK
    if mark == COMPUTER_MARK:
        ai_piece = AI_MARK

    # if the player has 4 pieces adjacent in the window, then he won
    # so we add a large score to its score, because he won
    if window.count(mark) == 4:
        score += 100
    # of the player has one move left to win, we maximize his score to put the forth piece
    # in this window to win
    elif window.count(mark) == 3 and window.count(EMPTY) == 1:
        score += 20
    elif window.count(mark) == 2 and window.count(EMPTY) == 2:
        score += 5

    # if the opponent has one move left to win, we minimize his score to prevent his winning
    if window.count(ai_piece) == 3 and window.count(EMPTY) == 1:
        score -= 80
    # elif window.count(ai_piece) == 2 and window.count(EMPTY) == 2:
    #     score -= 5
    return score


# function to loop over the board and check the score of each window in it
def window_score(board, mark):
    score = 0 # intializing the score by zero
    center_array = [] # an array to store the center column
    for i in list(board[:, COLNUM // 2]):
        center_array.append(int(i))

    # Multiply the score of the center col by anu number to increase the probability of winning
    score += center_array.count(mark) * 3

    # Iterates over each row, then each col to calculate the score of each window and 
    # store in score variable (check the score horizontaly)
    for r in range(ROWNUM):
        row_array = [] # array to store all rows, to loop over each col to calculate the score of each window
        for i in list(board[r, :]):
            row_array.append(int(i))
        for c in range(COLNUM - 3):
            window = row_array[c: c + WINDOW_SIZE] # list to store the window starting from an index to the widow size
            score += calculateScore(window, mark)

    # Score Vertical
    for c in range(COLNUM):
        col_array = [] # array to store all cols, to loop over each col to calculate the score of each window
        for i in list(board[:, c]):
            col_array.append(int(i))
        for r in range(ROWNUM - 3):
            window = col_array[r: r + WINDOW_SIZE] # list to store the window starting from an index to the widow size
            score += calculateScore(window, mark)

    # Score positive sloped diagonal
    for r in range(ROWNUM - 3):
        for c in range(COLNUM - 3):
            window = []
            for i in range(WINDOW_SIZE):
                window.append(board[r + i][c + i]) # append each diagonal window to check its score

            score += calculateScore(window, mark)

    # Score negative sloped diagonal
    for r in range(ROWNUM - 3):
        for c in range(COLNUM - 3):
            window = []
            for i in range(WINDOW_SIZE):
                window.append(board[r + 3 - i][c + i]) # append each diagonal window to check its score
            score += calculateScore(window, mark)

    return score



# function to implement minimax using alpha beta
# This function recursively evaluates the possible moves from the current board state 
# using the minimax algorithm with alpha-beta pruning. It aims to find the optimal move for the maximizing player (AI agent) while considering the opponent's (computer) moves.
def minimaxAlphaBeta(board, depth, alpha, beta, maximizing_player):
    validCols = eachValidColumn(board) # list to store all valid columns
    # if we have reached depth = 0 (have reached the final states) or any one wins or it's a draw
    if depth == 0 or is_terminal_node(board):
        if is_terminal_node(board):
            if isWinning(board, AI_MARK): # we want to make the ai agent wins, so we return a large score
                return None, 100000000000000
            elif isWinning(board, COMPUTER_MARK): # if computer wins, then this is a lose to our priority
                return None, -10000000000000
            else:
                return None, 0 # if it's a draw game
        else:
            return None, window_score(board, AI_MARK) # if we haven't reached the end yest, we check the score of each window in the board

    # if the maximizing player is the AI_Agent
    if maximizing_player:
        bestScore = -math.inf
        colNum = random.choice(validCols) # select random col to start the search
        for col in validCols:
            row = findDeepestRow(board, col)
            tempBoard = copy.deepcopy(board)
            tempBoard[row][col] = AI_MARK
            score = minimaxAlphaBeta(tempBoard, depth - 1, alpha, beta, False)[1]
            if score > bestScore:
                bestScore = score
                colNum = col
            alpha = max(alpha, bestScore)
            if beta <= alpha:
                break
        return colNum, bestScore
    # If the maximizing player is the computer, it aims to minimize the score
    else:
        bestScore = math.inf
        colNum = random.choice(validCols)
        for col in validCols:
            row = findDeepestRow(board, col)
            tempBoard = copy.deepcopy(board)
            tempBoard[row][col] = COMPUTER_MARK
            score = minimaxAlphaBeta(tempBoard, depth - 1, alpha, beta, True)[1]
            if score < bestScore:
                bestScore = score
                colNum = col
            beta = min(beta, bestScore)
            if beta <= alpha:
                break
        return colNum, bestScore # the function returns the selected column and the best score.


# function to implement minimax using alpha beta
# this function aims to find the optimal move for the maximizing player while considering the opponent's moves.
def minimax(board, depth, maximizingPlayer):
    validCols = eachValidColumn(board)
    if depth == 0 or is_terminal_node(board):
        if is_terminal_node(board):
            if isWinning(board, AI_MARK):
               return None, 100000000000000
            elif is_terminal_node(board) :
              if isWinning(board, COMPUTER_MARK):
               return None, -10000000000000
            else :
             return None, 0
        else:
            return None, window_score(board, AI_MARK)

    # If the current player is the maximizing player (AI agent), 
    # randomly selects a valid column to start the search.
    if maximizingPlayer:
        bestScore = -math.inf  # it initializes the best score as negative infinity
        colNum = random.choice(validCols) # randomly selects a valid column to start the search.
        for col in validCols:
            row = findDeepestRow(board, col)
            tempBoard = board.copy()
            tempBoard[row][col] = AI_MARK
            score = minimax(tempBoard, depth - 1, False)[1]
            if score > bestScore:
                bestScore = score
                colNum = col
        return colNum, bestScore
    else:
        bestScore = math.inf
        colNum = random.choice(validCols)
        for col in validCols:
            row = findDeepestRow(board, col)
            tempBoard = board.copy()
            tempBoard[row][col] = AI_MARK
            score = minimax(tempBoard, depth - 1, True)[1]
            if score < bestScore:
                bestScore = score
                colNum = col

        return colNum, bestScore


def draw_board(board):
    for c in range(COLNUM):
        for r in range(ROWNUM):
            rect_x = c * EACH_PIECE_SIZE
            rect_y = r * EACH_PIECE_SIZE + EACH_PIECE_SIZE
            pygame.draw.rect(screen, BLUE, (rect_x, rect_y, EACH_PIECE_SIZE, EACH_PIECE_SIZE))
            
            circle_x = int(rect_x + EACH_PIECE_SIZE / 2)
            circle_y = int(rect_y + EACH_PIECE_SIZE / 2)
            pygame.draw.circle(screen, WHITE, (circle_x, circle_y), RADIUS)

    for c in range(COLNUM):
        for r in range(ROWNUM):
            if board[r][c] == COMPUTER_MARK:
                circle_x = int(c * EACH_PIECE_SIZE + EACH_PIECE_SIZE / 2)
                circle_y = height - int(r * EACH_PIECE_SIZE + EACH_PIECE_SIZE / 2)
                pygame.draw.circle(screen, RED, (circle_x, circle_y), RADIUS)
            elif board[r][c] == AI_MARK:
                circle_x = int(c * EACH_PIECE_SIZE + EACH_PIECE_SIZE / 2)
                circle_y = height - int(r * EACH_PIECE_SIZE + EACH_PIECE_SIZE / 2)
                pygame.draw.circle(screen, YELLOW, (circle_x, circle_y), RADIUS)

    pygame.display.update()



root.mainloop()