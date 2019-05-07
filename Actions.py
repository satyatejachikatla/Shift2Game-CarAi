'''
Defining required actions.
'''

import ActionKeys
from time import sleep

def straight(x):
	if x >= 0.5:
		ActionKeys.PressKey(ActionKeys.DIK_UP)
	else:
		ActionKeys.ReleaseKey(ActionKeys.DIK_UP)

def left(x):
	if x >= 0.5:
		ActionKeys.PressKey(ActionKeys.DIK_LEFT)
	else:
		ActionKeys.ReleaseKey(ActionKeys.DIK_LEFT)

def right(x):
	if x >= 0.5:
		ActionKeys.PressKey(ActionKeys.DIK_RIGHT)
	else:
		ActionKeys.ReleaseKey(ActionKeys.DIK_RIGHT)


'''
print('waiting')
sleep(10)
print('start')

straight()
'''