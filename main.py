import cv2
import numpy as np


class runall:

    def __init__(self, model):
        self.drawing = False
        self.img = np.zeros((784, 784), np.uint8)
        self.model = model
        print('======= INSTRUCTION =======')
        print('press `r` to reset drawing')
        print('press `esc` to close')
        print('======= INSTRUCTION =======')
        print('')

    # Draws on the window on mouse event
    def draw_data(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
        if event == cv2.EVENT_MOUSEMOVE:
            if self.drawing == True:
                cv2.circle(self.img, (x, y), 20, (255, 255, 255), -1)
        if event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            print("The predicted number is %d" % self.predict( cv2.resize(self.img, dsize=(28, 28)) ))

    def runnp(self):

        cv2.namedWindow('Draw Image')
        cv2.setMouseCallback('Draw Image', self.draw_data)

        while (1):
            cv2.imshow('Draw Image', self.img)
            cv2.imshow('Resized', cv2.resize(self.img, dsize=(28, 28)))
            k = cv2.waitKey(20) & 0xFF
            if k == 27:
                break
            elif k == ord('r'):
                # Resetting the Window to redraw/repredict
                self.img = np.zeros((784, 784), np.uint8)

        cv2.destroyAllWindows()

    def predict(self, nparray):
        '''
        Convert the cv2 array into model useable array and predict it's value
        :param nparray:
        :return: Preducted Value
        '''
        predicted = self.model.predict(nparray.reshape(1, 784))
        predicted = np.argmax(predicted)
        return predicted

    def compile(self):
        self.runnp()