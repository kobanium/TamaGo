import sys

def print_out(message):
    print(message)

def print_err(message):
    print(message, file=sys.stderr)
