#
# errors.py
# A. Bellamine, L-D. Azoulay, N. Berrebi
# Class for managing error messages
#

error_dictionary = {
    "invalid_board":"The chess board composition is invalid.",
    "missing_exception":"An unknown exception have been raised.",
    "incorrect_move":"The move is incorrect and cannot be played."
}

def raiseException(exception_code):
    if exception_code in error_dictionary.keys():
        raise Exception(error_dictionary[exception_code]) 
    else:
        raise Exception(error_dictionary["missing_exception"]) 