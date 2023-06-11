import tkinter

def run_CU0():
    countThreshold = var_countThreshold.get()
    countType = var_countType.get()
    text.delete(1.0, 'end')
    text.insert('end', str(countThreshold) + "\n" + str(countType))

win = tkinter.Tk()
win.title('Deep Learning Results')
width = 1600
height = 1000
screenwidth = win.winfo_screenwidth()
screenheight = win.winfo_screenheight()
root_str = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
win.geometry(root_str)
win.maxsize(1600, 1000)

tkinter.Label(win, text = 'ACTIONS => ').place(relx=0.0, rely=0.0, relwidth=0.1, relheight=0.1)
btnCU0 = tkinter.Button(win, text="RUN_COPPER", command=run_CU0)
btnCU0.place(relx=0.1, rely=0.0, relwidth=0.3, relheight=0.1)

tkinter.Label(win, text = 'COUNT THRESHOLD => ').place(relx=0.1, rely=0.1, relwidth=0.2, relheight=0.1)
var_countThreshold = tkinter.IntVar()
var_countThreshold.set(value=33)
entry_countThreshold = tkinter.Entry(win, textvariable=var_countThreshold, font=('Arial', 20))
entry_countThreshold.place(relx=0.3, rely=0.1, relwidth=0.1, relheight=0.1)

tkinter.Label(win, text = 'COUNT TYPE CHECKED => ').place(relx=0.5, rely=0.1, relwidth=0.2, relheight=0.1)
var_countType = tkinter.IntVar()
var_countType.set(value=0)
cBox = tkinter.Checkbutton(win, text="NEGATIVE", variable=var_countType, onvalue=1, offvalue=0, font=('Arial', 20))
cBox.place(relx=0.7, rely=0.1, relwidth=0.2, relheight=0.1)

text = tkinter.Text(win)
text.place(rely=0.2, relwidth=1, relheight=0.8)
win.mainloop()

print("DONE!")