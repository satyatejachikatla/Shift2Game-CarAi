# getkeys.py
# Citation: Box Of Hats (https://github.com/Box-Of-Hats )

# For list of keys for GetAsyncKeyState
# https://docs.microsoft.com/en-us/windows/desktop/inputdev/virtual-key-codes

'''
# Original Copied code from sentdex

import win32api as wapi
import time

keyList = ["\b"]
for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789,.'APS$/\\":
    keyList.append(char)

def key_check():
    keys = []
    for key in keyList:
        if wapi.GetAsyncKeyState(ord(key)):
            keys.append(key)
    return keys
'''


import win32api as wapi
import time

VK_LEFT     = 0x25 # Left Arrow Key
VK_UP       = 0x26 # Up Arrow Key
VK_RIGHT    = 0x27 # Right arrow Key
VK_DOWN     = 0x28 # Down arrow Key
VK_BACK     = 0x08 # BACKSPACE key

VK_RETURN   = 0x0D # Enter key
VK_ESCAPE   = 0x1B # ESC key


keyList = { VK_LEFT , VK_UP , VK_RIGHT , VK_DOWN , VK_BACK , VK_RETURN , VK_ESCAPE}

def key_check():
    keys = []
    for key in keyList:
        if wapi.GetAsyncKeyState(key):
            keys.append(key)
    return keys


if __name__=='__main__':
    while True:
        print(key_check())
