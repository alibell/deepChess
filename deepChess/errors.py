#
# errors.py
# A. Bellamine, L-D. Azoulay, N. Berrebi
# Class for managing error messages
#

error_dictionary = {
    "invalid_board":"The chess board composition is invalid.",
    "missing_exception":"An unknown exception have been raised.",
    "incorrect_move":"The move is incorrect and cannot be played.",
    "unknown_player":"The player should be 0 or 1.",
    "invalid_k_player":"The k parameter should be between 0 and 1."
}

def raiseException(exception_code, parameter = ""):
    if exception_code in error_dictionary.keys():
        raise Exception(f"{error_dictionary[exception_code]} - {parameter}") 
    else:
        raise Exception(error_dictionary["missing_exception"]) 