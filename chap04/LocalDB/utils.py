import threading
import numpy as np

class comparer(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        import face_recognition
        from collections import deque
        import dataloader
        # init value
        self.state = True
        self.images = deque()
        # load face encodins from data
        self.DB_face_encodings, self.DB_face_names = dataloader.load_encode_data()
        # load function
        self.FE = face_recognition.face_encodings
        self.CF = face_recognition.compare_faces
        self.FD = face_recognition.face_distance

    def run(self):
        while self.state:
            if self.images:
                face_encoding = self.FE(**self.images.popleft())
                if face_encoding:
                    ret = self.CF(self.DB_face_encodings, face_encoding[0],tolerance=0.6)
                    if True in ret:
                        print(self.DB_face_names[ret.index(True)])
                    # self.comare_faces(face_encoding[0])

    def push(self,image):
        self.images.append(image)

    def exit(self):
        self.state = False

    def is_busy(self):
        return len(self.images) > 2

    def comare_faces(self, face_encoding_to_check, tolerance=0.95):
        """ don't Use now... """
        for idx, known_face_encodings in enumerate(self.DB_face_encodings):
            if len(known_face_encodings) == 0:
                np.empty((0))
            # np.mean(np.linalg.norm(known_face_encodings - face_encoding_to_check, axis=1))
            if np.mean(np.dot(known_face_encodings,face_encoding_to_check) /
                       (np.linalg.norm(known_face_encodings)*np.linalg.norm(face_encoding_to_check))) > tolerance:
                # Using cos distance
                print(self.DB_face_names[idx])

