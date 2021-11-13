import cv2, sys, numpy, os

size = 4
haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'datasets'

print('Reconociendo el rostro por favor asegurese de estar en un lugar iluminado')

(images, labels, names, id) = ([], [], [], 0)
# face recognizer

for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(datasets, subdirs)
        for filename in os.listdir(subjectpath):
            path = subjectpath + '/' + filename
            label = id
            images.append(cv2.imread(path, 0))
            labels.append(int(label))
        id += 1
(width, height) = (130, 100)

(images, labels) = [numpy.array(lis) for lis in [images, labels]]

# opencv trains a model from images
model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, labels)

face_cascade = cv2.CascadeClasifier(haar_file)
webcam = cv2.VideoCapture(0)  # selects the laptop camera

while True:
    (_, im) = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiscale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))
        prediction = model.predict(face_resize)
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)

        if prediction[1] < 500:
            cv2.putText(im, '% s - %.of' %
                        (names[prediction[0]], prediction[1]), (x - 10, y - 10),
                        cv2.FRONT_HERSHEY_PLAIN, 1, (0, 255, 0))

        else:
            cv2.putText(im, 'No reconocido',
                        (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
            cv2.imshow('OpenCV', im)
            key = cv2.waitKey(10)
            if key == 27:
                break
