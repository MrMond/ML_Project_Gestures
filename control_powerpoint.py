import comtypes.client as pptx_client
import pyautogui

# TODO bugfixing & testing

class PowerPoint:
    def __init__(self,path:str):
        self.client = None
        self.presentation = None
        self.load_powerpoint(path)

    def load_powerpoint(self,path:str):
        self.client = pptx_client.CreateObject("PowerPoint.Application")
        self.client.Visible = True

        pres = self.client.Presentations.open(path)
        self.presentation = pres.SlideShowSettings.Run()
    
    def advance_slide(self):
        if self.client:
            self.presentation.View.Next()

    def return_slide(self):
        if self.client:
            self.presentation.View.Previous()

    def toggle_blacken(self):
        if self.client:
            pyautogui.press('b') # toggle black screen

    # allow "with" syntax:

    def __enter__(self,path:str):
        self.load_powerpoint(path)

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.presentation.View.Exit()
        except:
            pass
        self.client = None
        self.presentation = None
