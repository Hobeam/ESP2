import glob
import numpy as np

def preprocess():
    """
    preprocess function
    ./faces
        |- id001
        |   |- image001.jpg
        |   |- image002.jpg
        |   |-      ...
        |- id002
        |   |- image001.jpg
        |   |- image002.jpg
        |   |-      ...
        |- ...
    cal id's face_encodings [encode_image001.jpg, encode_image002.jpg, ...]
    cal mean id's identity e_vector axis=0 -> I think it's not self-evident -> hot fix
    save e_vector in data folder id001.dat, id002.dat, ...
    :return: len of id
    """
    import face_recognition
    from tqdm import tqdm
    from os import makedirs
    makedirs('faces', exist_ok=True)
    makedirs('data', exist_ok=True)
    faces = glob.glob('faces/*')
    if not faces:
        raise Exception('Warning::: NO FACE DATA')
    for path_face in faces:
        face_images_path = glob.glob(path_face+'/*')
        encodings = []
        for path in tqdm(face_images_path):
            image=face_recognition.load_image_file(path)
            e_vector = face_recognition.face_encodings(image)
            if e_vector:
                encodings.append(e_vector[0])
            else:
                print("Warning::: face not detected in {}.".format(path))

        id = np.mean(encodings, axis=0) # identity를 내포하고 있는가? 흠...
        # id = np.median(encodings, axis=0)
        # id = np.array(encodings)
        id.dump('data/' + path_face.split('/')[-1] + '.dat')
    return len(faces)


def load_encode_data():
    """
    load encoded data
    ./data
        |- id001.dat
        |- id002.dat
        |-      ...
    :return: [id's encode vector], [name of id]
    """
    encodes = glob.glob('data/*')
    e_vectors = []
    names = []
    for idx, path in enumerate(encodes):
        e_vectors.append(np.load(path, allow_pickle=True))
        names.append(path.split('/')[-1][:-4])
    return e_vectors, names


if __name__ == '__main__':
    print(preprocess(), 'member preprocessed')
