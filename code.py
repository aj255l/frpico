# type: ignore
#Modified version of the original Adafruit code, and educ8's guide: 
#https://github.com/adafruit/Adafruit_CircuitPython_PCD8544/blob/main/examples/pcd8544_simpletest.py

# UART Code taken from:
# https://learn.adafruit.com/uart-communication-between-two-circuitpython-boards/code

import time
import board
import busio
import digitalio, displayio
from analogio import AnalogIn
import adafruit_pcd8544
from adafruit_simplemath import map_range
from frp_pico import *
from presets import *

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
display.contrast = 50

# Joystick Pins
joystickVRX = AnalogIn(board.GP27)
joystickVRY = AnalogIn(board.GP26)
joystickSwitch = digitalio.DigitalInOut(board.GP22)
joystickSwitch.switch_to_input(pull=digitalio.Pull.UP)

# Mode and Setup Buttons
modeButton = digitalio.DigitalInOut(board.GP20) # Pin 26
observeButton = digitalio.DigitalInOut(board.GP21) # Pin 27
modeButton.switch_to_input(pull=digitalio.Pull.DOWN)
observeButton.switch_to_input(pull=digitalio.Pull.DOWN)

# UART - TX
"""
NOTE: The TX/RX wires need to be connected to *different* UART
outputs on the Pico (e.g. output on UART1, receive on UART0), otherwise
the outputs will interfere.
"""
uartTX = busio.UART(board.GP4, board.GP5, baudrate=9600, timeout=0)
uartRX = busio.UART(board.GP16, board.GP17, baudrate=9600, timeout=0)

#####################
# Mode Logic
#####################
MODE = "KIND"

#####################
# Other Globals
#####################
# Value received over UART
rcvdValue = None

# For now, we just create FRPs by hardcoding them here:
frp = FRP(uniform([1, 2]))

# Menu Screen
presetRow = 0
presets = [
    ("Uniform 1-10", uniform(list(range(1, 11)))),
    ("Coin Flip", uniform([1, 2])),
    ("Biased Coin", weighted([0, 1], [2, 1])),
    ("Biased Coin2", weighted([0, 1], [2, 1])),
    ("Biased Coin3", weighted([0, 1], [2, 1])),
    ("Biased Coin4", weighted([0, 1], [2, 1])),
    ("Biased Coin5", weighted([0, 1], [2, 1])),
    # ("Conditional", ConditionalKind({i : uniform([i, i + 1]) for i in range(1, 11)}))
]

# Mode selection screen
modeRow = 0
MENU_MODES = [
        "KIND",
        "VALUE",
        "CONTRAST",
        "PRESETS"
    ]

# Clear the display.  Always call show after changing pixels to make the display
# update visible!
display.fill(0)
display.show()

def modeKind():
    observedText = "OBSERVED" if frp.isObserved() else "UNOBSERVED" 
    display.text(observedText, 0, 0, 1)

    lines = frp.display().splitlines()

    displayRow = 0
    for rowIdx in range(frp.kind.row, len(lines)):
        row = lines[rowIdx][frp.kind.col:frp.kind.cols]
        textY = (displayRow + 1) * 8
        display.text(row, 0, textY, 1)
        
        displayRow += 1

    display.show()


def getJoystickDirection():
    xValue = map_range(joystickVRX.value, 150, 65536, 0, 255)
    yValue = map_range(joystickVRY.value, 150, 65536, 0, 255)

    # Some directions - I don't know if they make sense
    if xValue <= 10:
        return "DOWN"
    elif xValue >= 240:
        return "UP"
    elif yValue <= 10:
        return "LEFT"
    elif yValue >= 240:
        return "RIGHT"

def doJoystick():
    direction = getJoystickDirection()
    
    if MODE == "KIND":
        if direction == "DOWN":
            frp.kind.scrollDown()
        elif direction == "UP":
            frp.kind.scrollUp()
        elif direction == "LEFT":
            frp.kind.scrollLeft()
        elif direction == "RIGHT":
            frp.kind.scrollRight()
    elif MODE == "PRESETS":
        global presetRow
        if direction == "DOWN":
            presetRow = min(len(presets) - 1, presetRow + 1)
        elif direction == "UP":
            presetRow = max(0, presetRow - 1)
    elif MODE == "CONTRAST":
        if direction == "UP":
            display.contrast = min(70, display.contrast + 1)
        elif direction == "DOWN":
            display.contrast = max(0, display.contrast - 1)
    elif MODE == "MENU":
        global modeRow
        if direction == "DOWN":
            modeRow = min(len(MENU_MODES) - 1, modeRow + 1)
        elif direction == "UP":
            modeRow = max(0, modeRow - 1)

def displayWrappedText(text, startRow=0):
    if not isinstance(text, str):
        text = str(text)
    displayRow = startRow
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
    if not frp.isObserved():
        displayWrappedText("Nothing to see here. FRP is UNOBSERVED.")
        return

    displayWrappedText(frp.getObserved())

"""
TEMPORARY: Display the received value on the screen
"""
def modeRCVD():
    display.text("Received :-)", 0, 0, 1)
    displayWrappedText(rcvdValue, startRow=1)

"""
Menu Screen
"""
def modePreset():
    display.text("Presets".center(15), 0, 0, 1)
    display.line(0, 8, 84, 8, 1)

    startY = 10
    fontHeight = 8
    for i in range(4): # Can display 4 at once
        displayX = 0
        displayY = startY + fontHeight*i

        presetIdx = presetRow + i
        if presetIdx >= len(presets):
            # Out of range
            break

        presetName, _ = presets[presetIdx]
        if presetIdx == presetRow:
            presetName = "> " + presetName
        display.text(presetName, displayX, displayY, 1)

    display.show()

"""
Contrast Screen
"""
def modeContrast():
    display.text("Contrast".center(15), 0, 0, 1)
    display.line(0, 8, 84, 8, 1)

    display.text(str(display.contrast), 0, 10, 1)

    display.show()
    
"""
Mode Select Screen
"""
def modeMenu():
    display.text("Menu".center(15), 0, 0, 1)
    display.line(0, 8, 84, 8, 1)

    startY = 10
    fontHeight = 8
    for i in range(4): # Can display 4 at once
        displayX = 0
        displayY = startY + fontHeight*i

        modeIdx = modeRow + i
        if modeIdx >= len(MENU_MODES):
            # Out of range
            break

        modeName = MENU_MODES[modeIdx]
        modeName = modeName[0] + modeName[1:].lower()
        if modeIdx == modeRow:
            modeName = "> " + modeName 
        display.text(modeName, displayX, displayY, 1)

    display.show()

    
jumpTable = {
    "KIND": modeKind,
    "VALUE": modeValue,
    "RCVD": modeRCVD,
    "MENU": modeMenu,
    "CONTRAST": modeContrast,
    "PRESETS": modePreset,
}

def switchMode(mode):
    display.fill(0)
    global MODE
    MODE = mode
    jumpTable[MODE]()

#########################
# LOOP VARIABLES
#########################
UPDATE_INTERVAL = 3.0
last_time_sent = 0

# Wait for the beginning of a message.
message_started = False

prevModeButtonState = modeButton.value
prevObserveButtonState = observeButton.value
prevJoystickSwitchState = joystickSwitch.value
while True:
    ##########################
    # Loop Variables
    ##########################
    modeButtonState = modeButton.value
    observeButtonState = observeButton.value
    joystickSwitchState = joystickSwitch.value
    now = time.monotonic()

    jumpTable[MODE]()
    
    doJoystick()

    # Handle Button Presses
    if modeButtonState != prevModeButtonState and modeButtonState:
        # if MODE == "KIND":
        #     switchMode("VALUE")
        # elif MODE == "VALUE":
        #     switchMode("MENU")
        # elif MODE == "MENU":
        #     switchMode("CONTRAST")
        # elif MODE == "CONTRAST":
        #     switchMode("KIND")
        modeRow = 0
        switchMode("MENU")

    elif observeButtonState != prevObserveButtonState and observeButtonState:
        frp.observe()
        switchMode("VALUE")
    elif joystickSwitchState != prevJoystickSwitchState and (not joystickSwitchState):
        # Reset FRP
        if MODE == "KIND" and frp.isObserved():
            frp.observed = None

        # Select on the menu screen
        if MODE == "PRESETS":
            kind = presets[presetRow][1]
            if isinstance(kind, ConditionalKind):
                frp = ConditionalFRP(kind)
            else:
                frp = FRP(kind)
            switchMode("KIND")
        elif MODE == "MENU":
            switchMode(MENU_MODES[modeRow])


    # # TEMPORARY: Received Value Display
    if rcvdValue != None:
        switchMode("RCVD")
    if rcvdValue != None and isinstance(frp, ConditionalFRP):
        frp.giveObserved(rcvdValue)

    prevModeButtonState = modeButtonState
    prevObserveButtonState = observeButtonState
    prevJoystickSwitchState = joystickSwitchState
    
    time.sleep(0.1)
    display.fill(0)

    # UART Attempt - TX
    if now - last_time_sent >= UPDATE_INTERVAL and frp.isObserved():
        observedValue = frp.getObserved()
        if isinstance(observedValue, float):
            uartTX.write(bytes(f"<f,{observedValue}>", "ascii"))
        else:
            uartTX.write(bytes(f"<i,{observedValue}>", "ascii"))
        print(f"Transmitting observed value, {observedValue}")
        last_time_sent = now

    # UART - RX
    byte_read = uartRX.read(1)  # Read one byte over UART lines
    if not byte_read:
        # Nothing read.
        continue
    if byte_read == b"<":
        # Start of message. Start accumulating bytes, but don't record the "<".
        message = []
        message_started = True
        continue

    if message_started:
        if byte_read == b">":
            # End of message. Don't record the ">".
            # Now we have a complete message. Convert it to a string, and split it up.
            print(message)
            message_parts = "".join(message).split(",")
            message_type = message_parts[0]
            message_started = False

            if message[0] == 'i':
                rcvdValue = int("".join(message[2:]))
            else:
                rcvdValue = float("".join(message[2:]))
        else:
            # Accumulate message byte.
            message.append(chr(byte_read[0]))