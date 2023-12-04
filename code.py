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
import random
from frp_pico import FRP_Pico
from frplib_pico.kinds import uniform, weighted_as

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
frp = FRP_Pico(uniform([1, 2, 3]))

# Placeholder kinds
smallKind = FRP_Pico(weighted_as([3, 4, 5], weights=[2, 6, 1]))
bigKind = FRP_Pico(uniform(range(20)))

# Clear the display.  Always call show after changing pixels to make the display
# update visible!
display.fill(0)
display.show()

def modeKind():
    observedText = "OBSERVED" if frp.isObserved() else "UNOBSERVED" 
    display.text(observedText, 0, 0, 1)

    lines = frp.display()

    displayRow = 0
    for rowIdx in range(frp.kindRow, len(lines)):
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

def doJoystick():
    direction = getJoystickDirection()
    
    if MODE == "KIND":
        if direction == "DOWN":
            frp.scrollDown()
        elif direction == "UP":
            frp.scrollUp()
        elif direction == "LEFT":
            frp.scrollLeft()
        elif direction == "RIGHT":
            frp.scrollRight()

def displayWrappedText(text, startRow=0):
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
    
    
jumpTable = {
    "KIND": modeKind,
    "VALUE": modeValue,
    "RCVD": modeRCVD
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
while True:
    ##########################
    # Loop Variables
    ##########################
    modeButtonState = modeButton.value
    observeButtonState = observeButton.value
    now = time.monotonic()

    jumpTable[MODE]()
    
    doJoystick()

    # Handle Button Presses
    if modeButtonState != prevModeButtonState and modeButtonState:
        if MODE == "KIND":
            switchMode("VALUE")
        else:
            switchMode("KIND")
    elif observeButtonState != prevObserveButtonState and observeButtonState:
        frp.observe()
        switchMode("VALUE")

    # TEMPORARY: Received Value Display
    if rcvdValue != None:
        switchMode("RCVD")
    
    time.sleep(0.1)
    display.fill(0)

    # UART Attempt - TX
    if now - last_time_sent >= UPDATE_INTERVAL and frp.isObserved:
        uartTX.write(bytes(f"<v,{frp.getObserved()}>", "ascii"))
        print("Transmitting observed value")
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

            rcvdValue = "".join(message[2:])
        else:
            # Accumulate message byte.
            message.append(chr(byte_read[0]))

    prevModeButtonState = modeButtonState
