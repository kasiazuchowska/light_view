import matplotlib.pyplot as plt

layout = [
    [sg.Button("Generate plot")],
    [sg.Image(key="plot")]
]

window = sg.Window("Plot generator", layout)

while True:
    event, values = window.read()
    if event == sg.WINDOW_CLOSED:
        break

    if event == "Generate plot":
        plt.plot([1,2,3], [1,4,9])
        plt.savefig("plot.png")
        window["plot"].update("plot.png")

window.close()
