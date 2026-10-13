def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
    """
    Transpose a 2D matrix by swapping rows and columns.
    
    Args:
        a: A 2D matrix of shape (m, n)
    
    Returns:
        The transposed matrix of shape (n, m)
    """
    if not a:
        return []
    rows,cols=len(a),len(a[0])
    for row in a:
        if len(row)!=cols:
            raise ValueError("maxtrix must be retangle matrix!")
    res=[]
    for col in range(cols):
        new_row=[]
        for row in range(rows):
            new_row.append(a[row][col])
        res.append(new_row)
    return res
            

    