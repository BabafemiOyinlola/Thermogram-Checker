from tkinter import *
from thermogramClassifier import ThermogramClassifier
import time
import cv2
from tkinter import filedialog as tkFileDialog

thermogram_classifier = ThermogramClassifier()

class App:

    def __init__(self, master):
        self.master = master
        self.frame = Frame(self.master)
        # self.master = master

        self.select_image_label = Label(master, text=self.select_image_text())
        self.select_image_label.grid(pady=10, ipady=5)
        # self.select_image_label.place(x=5, y=0, width=150)

        self.image_link = Entry(master, bd=3)
        self.image_link.place(x=30, y=50, width=300)

        self.select_image_button = Button(master, text="Browse Image", command=self.select_button)
        self.select_image_button.grid(row=1, column=1)

        self.processimage_button = Button(master, text="Process", command=self.process_image_button)
        self.processimage_button.grid(row=2, column=1, pady=10)

        self.result = Text(master, width=31, height=10, state="disabled")
        self.result.grid(sticky=E, pady=100)
        self.advice = Text(master, width=31, height=10, state="disabled")
        self.advice.grid(sticky=W, row=3, column=1, padx=0, pady=100)

        self.displaygraph_button = Button(master, text="Display Graph", state=DISABLED, command=self.display_graph_button)
        self.displaygraph_button.grid(row=4, column=1, pady=(0, 0), padx=165, sticky=(N, E, W))

    def select_button(self):
        print("Select a thermogram to check")
        img = tkFileDialog.askopenfile(parent=window, mode='rb', title='Please select an image')
        self.image_link.delete(0, END)
        self.image_link.insert(0, img.name)
        print(self.image_link.get())

    def select_image_text(self):
        return "Insert/browse link to thermogram"

    def process_image_button(self):
        imagelink = self.image_link.get()
        print("Processing.....")
        print(imagelink)

        result = self.main_method(imagelink)
        self.result.configure(state='normal')
        self.advice.configure(state='normal')
        if result == "NORMAL DATA":
            self.result.insert(INSERT, '\t\t\t\t\t RESULT' + '\n' + result)
            recommendation = "Thermogram is normal\nPlease continue to live healthy while carrying out medical checkups frequently"
            self.advice.insert(INSERT, '\t\t\t\t\t RECOMMENDATION' + '\n' + recommendation)

        elif result == "ABNORMAL DATA":
            self.result.insert(INSERT, '\t\t\t\t\t RESULT' + '\n\n' + result)
            recommendation = "There is an anomaly in this thermogram\n\nPlease see a doctor urgently"
            self.advice.insert(INSERT, '\t\t\t\t\t RECOMMENDATION' + '\n' + recommendation)
            self.frame.update_idletasks()

        self.result.configure(state='disabled')
        self.advice.configure(state='disabled')

        time.sleep(2)

        self.displaygraph_button.config(state='normal')


    def display_graph_button(self):
        print("Processing graph.....")
        analysis_ui = Toplevel(window)
        # analysis_ui.configure(bg='wine')
        analysis_ui.geometry('350x300')
        analysis_ui.title('Result')
        Label(analysis_ui, text='Graph').pack(padx=10, pady=10)


    def main_method(self, new_image_link):
        thermogram_classifier.train_from_text_file()
        sample_image = cv2.imread(new_image_link)
        thermogram_is_cancerous = thermogram_classifier.is_cancerous(sample_image)

        if thermogram_is_cancerous:
            return 'ABNORMAL DATA'
        else:
            return 'NORMAL DATA'


# if __name__ == '__main__':
window = Tk()
window.title('Thermogram Checker')
window.geometry('500x513')
window.configure(bg='teal')
app = App(window)
# analysis_ui = Toplevel(window)
window.mainloop()
