# type: ignore
#Modified version of the original Adafruit code, and educ8's guide: 
#https://github.com/adafruit/Adafruit_CircuitPython_PCD8544/blob/main/examples/pcd8544_simpletest.py

import time
import board
import busio
import digitalio, displayio
from analogio import AnalogIn
import adafruit_pcd8544
from adafruit_simplemath import map_range
import random

########################
# SETUP
########################

displayio.release_displays()

mosi_pin = board.GP7
clk_pin = board.GP6

spi = busio.SPI(clock=clk_pin, MOSI=mosi_pin)
dc = digitalio.DigitalInOut(board.GP2) # data/command
cs = digitalio.DigitalInOut(board.GP0) # Chip select
reset = digitalio.DigitalInOut(board.GP1) # reset

display = adafruit_pcd8544.PCD8544(spi, dc, cs, reset)

# Turn on the backlight
backlight = digitalio.DigitalInOut(board.GP8)
backlight.switch_to_output()
backlight.value = True

display.bias = 5
display.contrast = 46

# Joystick Pins
joystickVRX = AnalogIn(board.GP27)
joystickVRY = AnalogIn(board.GP26)

# Mode and Setup Buttons
modeButton = digitalio.DigitalInOut(board.GP20) # Pin 26
observeButton = digitalio.DigitalInOut(board.GP21) # Pin 27
modeButton.switch_to_input(pull=digitalio.Pull.DOWN)
observeButton.switch_to_input(pull=digitalio.Pull.DOWN)

#####################
# Mode Logic
#####################
MODE = "KIND"

#####################
# Other Globals
#####################
OBSERVED = False
observedValue = None

# Kind Scrolling
kindRow = 0
kindCol = 0

# Placeholder kinds
smallKind = """───────────
 ,- 2/9 - 3
-+- 6/9 - 4
 `- 1/9 - 5
───────────
"""
bigKind = """───────────────┐
  ,- 0.05 - 0  │
  |- 0.05 - 1  │
  |- 0.05 - 2  │
  |- 0.05 - 3  │
  |- 0.05 - 4  │
  |- 0.05 - 5  │
  |- 0.05 - 6  │
  |- 0.05 - 7  │
  |- 0.05 - 8  │
  |- 0.05 - 9  │
 -|            │
  |- 0.05 - 10 │
  |- 0.05 - 11 │
  |- 0.05 - 12 │
  |- 0.05 - 13 │
  |- 0.05 - 14 │
  |- 0.05 - 15 │
  |- 0.05 - 16 │
  |- 0.05 - 17 │
  |- 0.05 - 18 │
  `- 0.05 - 19 │
───────────────┘
"""
KINDSTR = bigKind

# Clear the display.  Always call show after changing pixels to make the display
# update visible!
display.fill(0)
display.show()

def modeKind():
    kind = KINDSTR
    observedText = "OBSERVED" if OBSERVED else "UNOBSERVED" 
    display.text(observedText, 0, 0, 1)

    lines = list(kind.splitlines())

    displayRow = 0
    for rowIdx in range(kindRow, len(lines)):
        row = lines[rowIdx]
        textY = (displayRow + 1) * 8
        display.text(row, 0, textY, 1)
        
        displayRow += 1

    display.show()


def getJoystickDirection():
    xValue = map_range(joystickVRX.value, 150, 65536, 0, 255)
    yValue = map_range(joystickVRY.value, 150, 65536, 0, 255)

    # Some directions - I don't know if they make sense
    if xValue <= 10:
        return "LEFT"
    elif xValue >= 240:
        return "RIGHT"
    elif yValue <= 10:
        return "DOWN"
    elif yValue >= 240:
        return "UP"

def kindScrollDown():
    global kindRow
    kindRows = len(list(KINDSTR.splitlines()))    

    kindRow = min(kindRow + 1, kindRows)

def kindScrollUp():
    global kindRow
    kindRow = max(0, kindRow - 1)

def doJoystick():
    direction = getJoystickDirection()
    
    if MODE == "KIND":
        if direction == "DOWN":
            kindScrollDown()
        elif direction == "UP":
            kindScrollUp()


def displayWrappedText(text):
    displayRow = 0
    displayCol = 0
    for i in range(len(text)):
        if text[i] == "\n":
            displayRow += 1
            displayCol = 0
            continue

        textX = displayCol * 8
        textY = displayRow * 8
        display.text(text[i], textX, textY, 1)
        
        if displayCol >= 9:
            displayRow += 1
            displayCol = 0
        else:
            displayCol += 1
        

    display.show()


"""
Display the value of the observed FRP on the screen, or that it is unobserved.
"""
def modeValue():
    if not OBSERVED:
        displayWrappedText("Nothing to see here. FRP is UNOBSERVED.")
        return

    valueStr = str(observedValue)
    displayWrappedText(valueStr)

def observe():
    global OBSERVED
    global observedValue
    OBSERVED = True

    value = random.uniform(1, 10)
    observedValue = value
    switchMode("VALUE")
    
    
jumpTable = {
    "KIND": modeKind,
    "VALUE": modeValue
}

def switchMode(mode):
    display.fill(0)
    global MODE
    MODE = mode
    jumpTable[MODE]()

prevModeButtonState = modeButton.value
prevObserveButtonState = observeButton.value
while True:
    modeButtonState = modeButton.value
    observeButtonState = observeButton.value

    jumpTable[MODE]()
    
    doJoystick()

    # Handle Button Presses
    if modeButtonState != prevModeButtonState and modeButtonState:
        if MODE == "KIND":
            switchMode("VALUE")
        else:
            switchMode("KIND")
    elif observeButtonState != prevObserveButtonState and observeButtonState:
        observe()

    time.sleep(0.1)
    display.fill(0)

    prevModeButtonState = modeButtonState
