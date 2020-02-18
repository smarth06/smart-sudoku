'''
    @Author - Harnoor Singh
'''
class Sudoku(object):  
    def __init__(self,position):
        self.positon = position

    position=[0,0]

    def find_unassigned_cells(self, sudoku_grid):
        for r in range(9):
            for c in range(9):
                if(sudoku_grid[r][c] == 0):
                    self.position[0] = r
                    self.position[1] = c
                    return True
        return False
    
    def in_row(self, sudoku_grid, row, number): 
        for i in range(9): 
            if(sudoku_grid[row][i] == number): 
                return True
        return False

    def in_col(self, sudoku_grid, col, number): 
        for i in range(9): 
            if(sudoku_grid[i][col] == number): 
                return True
        return False

    def in_box(self, sudoku_grid, row ,col ,number): 
        for i in range(3): 
            for j in range(3): 
                if(sudoku_grid[i+row][j+col] == number): 
                    return True
        return False

    def is_safe(self, sudoku_grid, row, col, number):
        return not self.in_row(sudoku_grid, row, number) and not self.in_col(sudoku_grid,col,number) and not self.in_box(sudoku_grid,row - row%3,col - col%3,number) 


    def solve_sudoku(self, sudoku_grid):
        if(not self.find_unassigned_cells(sudoku_grid)):
            return True   
        r = self.position[0]
        c = self.position[1]
        for i in range(1,10):
            if(self.is_safe(sudoku_grid,r,c,i)):
                sudoku_grid[r][c] = i

                if(self.solve_sudoku(sudoku_grid)):
                    return True

                sudoku_grid[r][c] = 0
        return False

if __name__=="__main__":
    sudoku_grid = [[0 for r in range(9)] for c in range(9)]
    position = [0,0]
    sudoku_grid=[[3,0,6,5,0,8,4,0,0], 
    [5,2,0,0,0,0,0,0,0], 
    [0,8,7,0,0,0,0,3,1], 
    [0,0,3,0,1,0,0,8,0], 
    [9,0,0,8,6,3,0,0,5], 
    [0,5,0,0,9,0,6,0,0], 
    [1,3,0,0,0,0,2,5,0], 
    [0,0,0,0,0,0,0,7,4], 
    [0,0,5,2,0,6,3,0,0]]

    print("Sudoku before solving:-->")
    print(sudoku_grid)

    sudoku = Sudoku(position)
    print("Sudoku after solving:-->")
    if(sudoku.solve_sudoku(sudoku_grid)):
        print(sudoku_grid)
    else:
        print("There is no solution for this particular Sudoku!!")