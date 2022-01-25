import cv2
import base64
import time
import datetime as dt
import csv

ORIGIN_PATH = "D:/uit/IE212 - Bigdata/final-project"

class Client():
    def __init__(self,
                 interval=3,
                 source="",
                 path_output = "",):
        self.interval = interval
        self.source = source
        self.path_output = path_output

        print('-'*50)
        print(f'Initialized data from {source}.')

        self.solve_video_to_frame()

    def solve_video_to_frame(self):
        """
            Tiến hành xử lý video, với mỗi giây trong video 
            nhóm sẽ lấy 1 khung hình để tiến hành xử lý
        """
        vidcap = cv2.VideoCapture(self.source)
        success, image = vidcap.read()

        fps = int(vidcap.get(cv2.CAP_PROP_FPS))
        frames = int(vidcap. get(cv2.CAP_PROP_FRAME_COUNT))
        seconds = 1 
        multiplier = int(fps * seconds)
        print(f'Fps: {fps}.')
        print(f'Duration : {int(frames/fps)}s')

        images = []
        while success:
            frameId = int(round(vidcap.get(1))) - 1
            success, image = vidcap.read()
            if frameId % multiplier == 0:
                images.append(image)
                print(f'init....')

            if len(images) >= 20:
                break
        vidcap.release()
        self.images = images
        print(f'Initialized completed!')
        print('-'*50)
        print('\n')


    def stream_video(self):
        
        if self.interval <= 0:
            print(f'Start send data.')
        else:
            print(f'Start send data every {self.interval} seconds')


        length = len(self.images)
        for idx,image in enumerate(self.images):

            if self.interval <= 0:
                input(f"Press Enter to send data ...")
            
            png = cv2.imencode('.png', image)[1]
            png_as_text = base64.b64encode(png).decode('utf-8')
            timestamp = dt.datetime.utcnow().isoformat()

            src = self.source.split('/')
            data = [png_as_text, timestamp, src[-1], idx]

            print(f'Sent image {idx + 1}/{length} to {self.path_output}')
            self.send_data(idx, data)
            
            if self.interval > 0:
                time.sleep(self.interval)
        
        print(f'Success!!!')

    def send_data(self, idx, data):
        with open(self.path_output + '/source_{:05n}.csv'.format(idx), 'w') as csv_file:
            wr = csv.writer(csv_file, delimiter= ",")
            wr.writerow(data)

if __name__ == '__main__':
    client = Client(interval=20,
                    source= ORIGIN_PATH + '/videos/test_2.mp4',
                    path_output= ORIGIN_PATH + '/streaming/input')

    client.stream_video()