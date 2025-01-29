from tkinter import *
import numpy as np
from PIL import ImageGrab
from Prediction import predict

# Create main window
window = Tk()
window.title("Handwritten Digit Recognition")
window.configure(bg="#f0f0f0")

# Customizing the font style
title_font = ('Arial', 26, 'bold')
button_font = ('Arial', 16)
result_font = ('Arial', 18, 'bold')

# Create label for result
l1 = Label(window, font=result_font, fg="green", bg="#f0f0f0")
l1.pack(pady=20)

# Function to handle image capture and prediction
def MyProject():
    global l1
    widget = cv
    x = window.winfo_rootx() + widget.winfo_x()
    y = window.winfo_rooty() + widget.winfo_y()
    x1 = x + widget.winfo_width()
    y1 = y + widget.winfo_height()

    # Image capture and resizing
    img = ImageGrab.grab().crop((x, y, x1, y1)).resize((28, 28))
    img = img.convert('L')  # Convert to grayscale

    # Convert image to vector
    x = np.asarray(img)
    vec = np.zeros((1, 784))
    k = 0
    for i in range(28):
        for j in range(28):
            vec[0][k] = x[i][j]
            k += 1

    # Load pre-trained model parameters
    Theta1 = np.loadtxt('Theta1.txt')
    Theta2 = np.loadtxt('Theta2.txt')

    # Prediction using neural network
    pred = predict(Theta1, Theta2, vec / 255)

    # Display prediction result
    l1.config(text="Predicted Digit: " + str(pred[0]))

# Initialize drawing variables
lastx, lasty = None, None

# Clear the canvas
def clear_widget():
    global cv, l1
    cv.delete("all")
    l1.config(text="")

# Activate drawing on canvas
def event_activation(event):
    global lastx, lasty
    cv.bind('<B1-Motion>', draw_lines)
    lastx, lasty = event.x, event.y

# Draw lines on canvas
def draw_lines(event):
    global lastx, lasty
    x, y = event.x, event.y
    cv.create_line((lastx, lasty, x, y), width=30, fill='black', capstyle=ROUND, smooth=True, splinesteps=12)
    lastx, lasty = x, y

# Title label with attractive font and positioning
L1 = Label(window, text="Handwritten Digit Recognition", font=title_font, fg="blue", bg="#f0f0f0")
L1.pack(pady=30)

# Buttons for actions
b1 = Button(window, text="Clear Canvas", font=button_font, bg="#ffcc00", fg="black", command=clear_widget, relief=SOLID, bd=2, width=20)
b1.pack(pady=10)

b2 = Button(window, text="Predict Digit", font=button_font, bg="#ff4d4d", fg="white", command=MyProject, relief=SOLID, bd=2, width=20)
b2.pack(pady=10)

# Canvas for drawing the digits
cv = Canvas(window, width=350, height=290, bg='#dcdcdc', bd=2, relief=SOLID)
cv.pack(pady=20)

# Bind mouse click for drawing
cv.bind('<Button-1>', event_activation)

# Set window dimensions
window.geometry("600x650")

# Run the Tkinter event loop
window.mainloop()
