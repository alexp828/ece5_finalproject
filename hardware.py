from pyfirmata import Arduino, util

import time

# red LED on 10
# green LED on 9
# buzzer on 3
# button switch on 5

REDLED = 10
GREENLED = 9
BUZZER = 3
BUTTON = 5

board = Arduino('/dev/cu.usbmodem141201') #Change this to your port


PIEZO_PIN = board.get_pin('d:' + str(BUZZER) + ':p') #digital output from 0-1

#iterator for input, so the program would keep updating
it = util.Iterator(board)
it.start()

switch = board.get_pin('d:' + str(BUTTON) + ':i') #digital input

keycode = 1 #Comment this out

# Our secret code = 0
secretCode = 0

on = False #Initialize button state

while True:
    buttonState = switch.read()
    if buttonState == True:
        if on == True:
            on = False
        else:
            on = True

    if keycode == secretCode:
        board.digital[GREENLED].write(1)
        if on == True:
            board.digital[GREENLED].write(0)
            board.exit()

    elif keycode != secretCode:
        PIEZO_PIN.write(0.5)
        board.digital[REDLED].write(1)
        if on == True:
            PIEZO_PIN.write(0)
            board.digital[REDLED].write(0)
            PIEZO_PIN.write(0)
            board.exit()
